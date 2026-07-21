"""Create (or update) a user in the users table with a bcrypt-hashed password.

No public signup flow exists; use this CLI to provision login accounts.

Usage:
  python -m scripts.create_user --username alice --password secret123
  python -m scripts.create_user --username alice --password newpass --update
"""
from __future__ import annotations
import argparse
import sys

from app.database import SessionLocal
from app.models import User
from app.security import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a user for login.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Reset the password if the user already exists",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == args.username).first()
        if existing:
            if not args.update:
                print(
                    f"User {args.username!r} already exists "
                    "(pass --update to reset the password)."
                )
                return 1
            existing.password_hash = hash_password(args.password)
            db.commit()
            print(f"Updated password for {args.username!r}.")
            return 0

        user = User(
            username=args.username, password_hash=hash_password(args.password)
        )
        db.add(user)
        db.commit()
        print(f"Created user {args.username!r}.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
