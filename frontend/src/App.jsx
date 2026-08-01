import { useState } from 'react'
import axios from 'axios'
import './App.css'

const initialFormData = {
  Gr_Liv_Area: 1500,
  Bedroom_AbvGr: 3,
  Year_Built: 2000,
  Full_Bath: 2
}

function App() {
  const [formData, setFormData] = useState(initialFormData)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: Number(value) })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData)
      const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
      }).format(response.data.predicted_price)

      setPrediction(formattedPrice)
    } catch (err) {
      console.error('Error fetching prediction:', err)
      setError('Unable to connect to the valuation service. Please ensure the API is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">AI-powered property insights</p>
          <h1>Estimate your home's value in seconds.</h1>
          <p className="hero-text">
            Enter a few property details and receive an instant valuation powered by your machine-learning model.
          </p>
          <div className="hero-badges">
            <span>Fast analysis</span>
            <span>Market-aware</span>
            <span>Instant result</span>
          </div>
        </div>

        <div className="hero-card">
          <h2>Property snapshot</h2>
          <div className="stats-grid">
            <div>
              <strong>Living area</strong>
              <span>{formData.Gr_Liv_Area} sq ft</span>
            </div>
            <div>
              <strong>Bedrooms</strong>
              <span>{formData.Bedroom_AbvGr}</span>
            </div>
            <div>
              <strong>Bathrooms</strong>
              <span>{formData.Full_Bath}</span>
            </div>
            <div>
              <strong>Built</strong>
              <span>{formData.Year_Built}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <form className="card form-card" onSubmit={handleSubmit}>
          <div className="card-header">
            <p className="section-label">Valuation form</p>
            <h2>Tell us about the property</h2>
          </div>

          <div className="form-grid">
            <label className="field">
              <span>Above Ground Living Area</span>
              <input type="number" name="Gr_Liv_Area" min="1" value={formData.Gr_Liv_Area} onChange={handleChange} />
            </label>

            <label className="field">
              <span>Number of Bedrooms</span>
              <input type="number" name="Bedroom_AbvGr" min="0" value={formData.Bedroom_AbvGr} onChange={handleChange} />
            </label>

            <label className="field">
              <span>Year Built</span>
              <input type="number" name="Year_Built" min="1800" value={formData.Year_Built} onChange={handleChange} />
            </label>

            <label className="field">
              <span>Full Bathrooms</span>
              <input type="number" name="Full_Bath" min="0" value={formData.Full_Bath} onChange={handleChange} />
            </label>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing market...' : 'Predict property value'}
          </button>
        </form>

        <aside className="card result-card">
          <div className="card-header">
            <p className="section-label">Smart estimate</p>
            <h2>See your projected value</h2>
          </div>

          <div className="result-body">
            {loading ? (
              <div className="result-loading">
                <div className="spinner" />
                <p>Running valuation model...</p>
              </div>
            ) : prediction ? (
              <>
                <p className="result-value">{prediction}</p>
                <p className="result-copy">
                  This estimate is based on your property profile and recent market patterns.
                </p>
              </>
            ) : error ? (
              <p className="result-error">{error}</p>
            ) : (
              <p className="result-placeholder">
                Your valuation will appear here after you submit the form.
              </p>
            )}
          </div>
        </aside>
      </section>
    </div>
  )
}

export default App