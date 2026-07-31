import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [formData, setFormData] = useState({
    Gr_Liv_Area: 1500,
    Bedroom_AbvGr: 3,
    Year_Built: 2000,
    Full_Bath: 2
  })
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: Number(e.target.value) })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      // This calls your Python API!
      const response = await axios.post('http://127.0.0.1:8000/predict', formData)
      
      // Format the number as currency
      const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
      }).format(response.data.predicted_price)
      
      setPrediction(formattedPrice)
    } catch (error) {
      console.error("Error fetching prediction:", error)
      setPrediction("Error connecting to AI")
    }
    setLoading(false)
  }

  return (
    <div className="App" style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Advanced Real Estate Forecaster</h1>
      <p>Enter property details below to get an instant AI valuation.</p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '2rem' }}>
        
        <label>
          Above Ground Living Area (sq ft):
          <input type="number" name="Gr_Liv_Area" value={formData.Gr_Liv_Area} onChange={handleChange} style={{ marginLeft: '10px' }} />
        </label>

        <label>
          Number of Bedrooms:
          <input type="number" name="Bedroom_AbvGr" value={formData.Bedroom_AbvGr} onChange={handleChange} style={{ marginLeft: '10px' }} />
        </label>

        <label>
          Year Built:
          <input type="number" name="Year_Built" value={formData.Year_Built} onChange={handleChange} style={{ marginLeft: '10px' }} />
        </label>

        <label>
          Full Bathrooms:
          <input type="number" name="Full_Bath" value={formData.Full_Bath} onChange={handleChange} style={{ marginLeft: '10px' }} />
        </label>

        <button type="submit" disabled={loading} style={{ padding: '10px', fontSize: '1rem', cursor: 'pointer', marginTop: '1rem' }}>
          {loading ? 'Analyzing Market...' : 'Predict Property Value'}
        </button>
      </form>

      {prediction && (
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#e0f7fa', borderRadius: '8px' }}>
          <h2>Estimated Value: <span style={{ color: '#00695c' }}>{prediction}</span></h2>
        </div>
      )}
    </div>
  )
}

export default App