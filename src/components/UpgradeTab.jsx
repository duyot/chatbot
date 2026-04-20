import './UpgradeTab.css'

export default function UpgradeTab() {
  return (
    <div className="upgrade-tab" role="button" tabIndex={0} aria-label="Upgrade to Pro">
      <span className="upgrade-tab-text">Upgrade to Pro</span>
      <span className="upgrade-tab-star" aria-hidden="true">✦</span>
    </div>
  )
}
