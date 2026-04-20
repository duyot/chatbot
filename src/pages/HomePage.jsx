import { useNavigate } from 'react-router-dom'
import './HomePage.css'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="home-page">
      <div className="home-content">
        <div className="home-badge">AI-powered chat</div>
        <h1 className="home-logo">CHAT A.I+</h1>
        <p className="home-tagline">
          Ask anything, explore ideas, and get instant answers
          powered by advanced AI.
        </p>
        <div className="home-cta-group">
          <button className="home-cta" onClick={() => navigate('/chat')}>
            Start chat
          </button>
          <button className="home-cta-outline" onClick={() => navigate('/chat')}>
            Learn more
          </button>
        </div>
      </div>
    </div>
  )
}
