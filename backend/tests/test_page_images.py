"""Tests for preview page-image rendering (app/services/page_images.py)."""
import os

import fitz
import pytest
from PIL import Image

from app.config import settings
from app.services import page_images


DOC_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Point page_images at a throwaway upload root."""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    return tmp_path


def _make_pdf(path, pages=3):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i + 1} body text")
    doc.save(str(path))
    doc.close()
    return str(path)


# --- render_document_pages --------------------------------------------------

def test_render_writes_one_image_per_page(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)

    images = page_images.render_document_pages(pdf, DOC_ID)

    assert [img.page for img in images] == [1, 2, 3]
    for img in images:
        assert os.path.exists(img.path)
        assert img.width > 0 and img.height > 0
        # Portrait A4-ish default page from fitz.new_page()
        assert img.height > img.width


def test_render_names_files_zero_padded_in_configured_format(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)

    page_images.render_document_pages(pdf, DOC_ID)

    names = sorted(os.listdir(page_images.pages_dir(DOC_ID)))
    assert names == ["0001.webp", "0002.webp"]


def test_render_honours_dpi_setting(upload_dir, tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)

    monkeypatch.setattr(settings, "page_image_dpi", 72)
    low = page_images.render_document_pages(pdf, DOC_ID)[0]

    page_images.delete_page_images(DOC_ID)
    monkeypatch.setattr(settings, "page_image_dpi", 144)
    high = page_images.render_document_pages(pdf, DOC_ID)[0]

    assert high.width > low.width


def test_render_is_idempotent_and_reuses_existing_files(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)

    first = page_images.render_document_pages(pdf, DOC_ID)
    mtimes = {img.path: os.path.getmtime(img.path) for img in first}

    second = page_images.render_document_pages(pdf, DOC_ID)

    assert [i.page for i in second] == [i.page for i in first]
    # Reused, not re-encoded.
    for img in second:
        assert os.path.getmtime(img.path) == mtimes[img.path]


def test_render_replaces_a_corrupt_image(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    page_images.render_document_pages(pdf, DOC_ID)

    path = page_images.page_image_path(DOC_ID, 1)
    with open(path, "wb") as f:
        f.write(b"not an image")

    images = page_images.render_document_pages(pdf, DOC_ID)

    assert len(images) == 1
    with Image.open(path) as img:
        assert img.size == (images[0].width, images[0].height)


def test_render_leaves_no_temp_files(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)

    page_images.render_document_pages(pdf, DOC_ID)

    names = os.listdir(page_images.pages_dir(DOC_ID))
    assert not [n for n in names if n.endswith(".tmp")]


def test_render_skips_non_pdf(upload_dir, tmp_path):
    img_path = tmp_path / "scan.png"
    Image.new("RGB", (10, 10)).save(img_path)

    assert page_images.render_document_pages(str(img_path), DOC_ID) == []
    assert not os.path.exists(page_images.pages_dir(DOC_ID))


def test_render_skips_missing_source(upload_dir, tmp_path):
    assert page_images.render_document_pages(str(tmp_path / "gone.pdf"), DOC_ID) == []


def test_render_disabled_by_flag(upload_dir, tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    monkeypatch.setattr(settings, "page_images_enabled", False)

    assert page_images.render_document_pages(pdf, DOC_ID) == []


def test_render_survives_a_single_bad_page(upload_dir, tmp_path, monkeypatch):
    """One unrenderable page must not cost us the rest of the document."""
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
    real_write = page_images._write_pixmap
    calls = {"n": 0}

    def flaky(pix, path, fmt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encoder blew up")
        return real_write(pix, path, fmt)

    monkeypatch.setattr(page_images, "_write_pixmap", flaky)

    images = page_images.render_document_pages(pdf, DOC_ID)

    assert [img.page for img in images] == [1, 3]


def test_failed_page_leaves_no_partial_file(upload_dir, tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)

    def boom(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fitz.Pixmap, "pil_save", boom)

    assert page_images.render_document_pages(pdf, DOC_ID) == []
    assert os.listdir(page_images.pages_dir(DOC_ID)) == []


# --- formats ---------------------------------------------------------------

@pytest.mark.parametrize("fmt,ext,media", [
    ("webp", "webp", "image/webp"),
    ("jpg", "jpg", "image/jpeg"),
    ("jpeg", "jpg", "image/jpeg"),
    ("png", "png", "image/png"),
])
def test_render_supports_each_format(upload_dir, tmp_path, monkeypatch, fmt, ext, media):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    monkeypatch.setattr(settings, "page_image_format", fmt)

    images = page_images.render_document_pages(pdf, DOC_ID)

    assert images[0].path.endswith(f".{ext}")
    assert page_images.media_type() == media
    with Image.open(images[0].path) as img:
        assert img.size == (images[0].width, images[0].height)


def test_unknown_format_falls_back_to_webp(monkeypatch):
    monkeypatch.setattr(settings, "page_image_format", "tiff")
    assert page_images.image_format() == "webp"


# --- list / delete ---------------------------------------------------------

def test_list_page_images_is_ordered_and_ignores_foreign_files(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=12)
    page_images.render_document_pages(pdf, DOC_ID)

    out_dir = page_images.pages_dir(DOC_ID)
    open(os.path.join(out_dir, "notes.txt"), "w").close()
    open(os.path.join(out_dir, "0003.webp.tmp"), "w").close()

    listed = page_images.list_page_images(DOC_ID)

    # 12 pages proves ordering is numeric, not lexicographic (0010 vs 0002).
    assert [p.page for p in listed] == list(range(1, 13))


def test_list_page_images_empty_when_never_rendered(upload_dir):
    assert page_images.list_page_images(DOC_ID) == []


def test_list_page_images_ignores_other_formats(upload_dir, tmp_path, monkeypatch):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    monkeypatch.setattr(settings, "page_image_format", "png")
    page_images.render_document_pages(pdf, DOC_ID)

    monkeypatch.setattr(settings, "page_image_format", "webp")
    assert page_images.list_page_images(DOC_ID) == []


def test_delete_page_images(upload_dir, tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
    page_images.render_document_pages(pdf, DOC_ID)

    assert page_images.delete_page_images(DOC_ID) == 2
    assert page_images.list_page_images(DOC_ID) == []
    assert page_images.delete_page_images(DOC_ID) == 0
