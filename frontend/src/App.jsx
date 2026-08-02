import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend
} from 'recharts'
import {
  Home, TrendingUp, BarChart3, ShieldCheck, Zap, Layers, Activity, Building2, Layers3
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const initialFormData = {
  Gr_Liv_Area: 1500,
  Bedroom_AbvGr: 3,
  Year_Built: 2000,
  Full_Bath: 2
}

function App() {
  const [activeTab, setActiveTab] = useState('valuation') // 'valuation' | 'analytics'
  const [formData, setFormData] = useState(initialFormData)
  const [prediction, setPrediction] = useState(null)
  const [predictionDetails, setPredictionDetails] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Analytics data
  const [kpis, setKpis] = useState(null)
  const [trends, setTrends] = useState([])
  const [sizeBrackets, setSizeBrackets] = useState([])
  const [featureImportance, setFeatureImportance] = useState([])
  const [analyticsLoading, setAnalyticsLoading] = useState(false)

  useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    setAnalyticsLoading(true)
    try {
      const [kpiRes, trendsRes, bracketsRes, featuresRes] = await Promise.all([
        axios.get(`${API_BASE}/analytics/kpis`),
        axios.get(`${API_BASE}/analytics/trends`),
        axios.get(`${API_BASE}/analytics/price-vs-sqft`),
        axios.get(`${API_BASE}/analytics/feature-importance`)
      ])
      setKpis(kpiRes.data)
      setTrends(trendsRes.data)
      setSizeBrackets(bracketsRes.data)
      setFeatureImportance(featuresRes.data)
    } catch (err) {
      console.warn('Analytics API unavailable or server starting:', err.message)
    } finally {
      setAnalyticsLoading(false)
    }
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: value === '' ? '' : Number(value) })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setPrediction(null)
    setPredictionDetails(null)

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData)
      const price = response.data.predicted_price
      const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
      }).format(price)

      setPrediction(formattedPrice)
      setPredictionDetails({
        pricePerSqFt: Math.round(response.data.price_per_sqft || (price / formData.Gr_Liv_Area)),
        percentile: response.data.market_percentile || 50
      })
    } catch (err) {
      console.error('Error fetching prediction:', err)
      setError('Unable to connect to the valuation service. Please ensure the API is running at http://127.0.0.1:8000.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      {/* Navbar Header */}
      <header className="app-header">
        <div className="header-container">
          <div className="brand">
            <Building2 className="brand-icon" />
            <div className="brand-text">
              <span className="brand-name">RealEstate ML</span>
              <span className="brand-tag">Valuation & Market Insights</span>
            </div>
          </div>

          <nav className="nav-tabs">
            <button
              className={`nav-tab ${activeTab === 'valuation' ? 'active' : ''}`}
              onClick={() => setActiveTab('valuation')}
            >
              <Home size={18} />
              <span>Valuation Estimator</span>
            </button>
            <button
              className={`nav-tab ${activeTab === 'analytics' ? 'active' : ''}`}
              onClick={() => setActiveTab('analytics')}
            >
              <TrendingUp size={18} />
              <span>Market Analytics</span>
            </button>
          </nav>
        </div>
      </header>

      {/* KPI Overview Bar */}
      {kpis && (
        <section className="kpi-banner">
          <div className="kpi-card">
            <Activity className="kpi-icon" />
            <div>
              <span className="kpi-label">Dataset Sample Size</span>
              <span className="kpi-value">{kpis.total_records?.toLocaleString()} Properties</span>
            </div>
          </div>
          <div className="kpi-card">
            <BarChart3 className="kpi-icon" />
            <div>
              <span className="kpi-label">Average Market Price</span>
              <span className="kpi-value">${Math.round(kpis.avg_price)?.toLocaleString()}</span>
            </div>
          </div>
          <div className="kpi-card">
            <Layers className="kpi-icon" />
            <div>
              <span className="kpi-label">Avg Price / Sq Ft</span>
              <span className="kpi-value">${Math.round(kpis.avg_price_sqft)} / sq ft</span>
            </div>
          </div>
        </section>
      )}

      {/* TAB 1: VALUATION ESTIMATOR */}
      {activeTab === 'valuation' && (
        <div className="tab-content">
          <section className="hero-panel">
            <div className="hero-copy">
              <p className="eyebrow">AI-POWERED PROPERTY INSIGHTS</p>
              <h1>Estimate your home's value in seconds.</h1>
              <p className="hero-text">
                Enter your property dimensions and details to receive an instant machine-learning valuation powered by our trained XGBoost model.
              </p>
              <div className="hero-badges">
                <span><Zap size={14} /> Instant Prediction</span>
                <span><ShieldCheck size={14} /> Market-Calibrated</span>
                <span><Layers3 size={14} /> Feature Weights</span>
              </div>
            </div>

            <div className="hero-card">
              <h2>Property Snapshot</h2>
              <div className="stats-grid">
                <div>
                  <strong>Living area</strong>
                  <span>{formData.Gr_Liv_Area || 0} sq ft</span>
                </div>
                <div>
                  <strong>Bedrooms</strong>
                  <span>{formData.Bedroom_AbvGr || 0}</span>
                </div>
                <div>
                  <strong>Bathrooms</strong>
                  <span>{formData.Full_Bath || 0}</span>
                </div>
                <div>
                  <strong>Year Built</strong>
                  <span>{formData.Year_Built || 0}</span>
                </div>
              </div>
            </div>
          </section>

          <section className="content-grid">
            <form className="card form-card" onSubmit={handleSubmit}>
              <div className="card-header">
                <p className="section-label">PROPERTY PARAMETERS</p>
                <h2>Tell us about the property</h2>
              </div>

              <div className="form-grid">
                <label htmlFor="Gr_Liv_Area" className="field">
                  <span>Above Ground Living Area (sq ft)</span>
                  <input
                    id="Gr_Liv_Area"
                    type="number"
                    name="Gr_Liv_Area"
                    min="1"
                    value={formData.Gr_Liv_Area}
                    onChange={handleChange}
                  />
                </label>

                <label htmlFor="Bedroom_AbvGr" className="field">
                  <span>Number of Bedrooms</span>
                  <input
                    id="Bedroom_AbvGr"
                    type="number"
                    name="Bedroom_AbvGr"
                    min="0"
                    value={formData.Bedroom_AbvGr}
                    onChange={handleChange}
                  />
                </label>

                <label htmlFor="Year_Built" className="field">
                  <span>Year Built</span>
                  <input
                    id="Year_Built"
                    type="number"
                    name="Year_Built"
                    min="1800"
                    max="2026"
                    value={formData.Year_Built}
                    onChange={handleChange}
                  />
                </label>

                <label htmlFor="Full_Bath" className="field">
                  <span>Full Bathrooms</span>
                  <input
                    id="Full_Bath"
                    type="number"
                    name="Full_Bath"
                    min="0"
                    value={formData.Full_Bath}
                    onChange={handleChange}
                  />
                </label>
              </div>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Calculating Valuation...' : 'Predict Property Value'}
              </button>
            </form>

            <aside className="card result-card">
              <div className="card-header">
                <p className="section-label">SMART ESTIMATE</p>
                <h2>See your projected value</h2>
              </div>

              <div className="result-body">
                {loading ? (
                  <div className="result-loading">
                    <div className="spinner" />
                    <p>Evaluating property parameters with XGBoost model...</p>
                  </div>
                ) : prediction ? (
                  <div className="result-success">
                    <span className="result-tag">Estimated Market Price</span>
                    <h3 className="result-price">{prediction}</h3>

                    {predictionDetails && (
                      <div className="result-metrics">
                        <div className="metric-pill">
                          <span className="metric-title">Price / Sq Ft</span>
                          <span className="metric-val">${predictionDetails.pricePerSqFt}</span>
                        </div>
                        <div className="metric-pill">
                          <span className="metric-title">Market Rank</span>
                          <span className="metric-val">{predictionDetails.percentile}th percentile</span>
                        </div>
                      </div>
                    )}

                    <p className="result-copy">
                      This estimate is based on your property profile and recent market transactions in the Ames dataset.
                    </p>
                  </div>
                ) : error ? (
                  <p className="result-error">{error}</p>
                ) : (
                  <p className="result-placeholder">
                    Fill out the property parameters and click <strong>Predict property value</strong> to see the valuation.
                  </p>
                )}
              </div>
            </aside>
          </section>
        </div>
      )}

      {/* TAB 2: MARKET ANALYTICS DASHBOARD */}
      {activeTab === 'analytics' && (
        <div className="tab-content analytics-tab">
          <div className="analytics-header">
            <h2>Market Trends & Model Feature Analysis</h2>
            <p>Explore historical price trajectories and machine-learning feature importance across the dataset.</p>
          </div>

          <div className="charts-grid">
            {/* Chart 1: Price Trends by Era */}
            <div className="card chart-card">
              <div className="chart-header">
                <div>
                  <p className="section-label">HISTORICAL TRAJECTORY</p>
                  <h3>Average Sale Price by Construction Decade</h3>
                </div>
              </div>

              <div className="chart-wrapper">
                {trends.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={trends} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="decade" stroke="#64748b" />
                      <YAxis
                        stroke="#64748b"
                        tickFormatter={(val) => `$${val / 1000}k`}
                      />
                      <Tooltip
                        formatter={(val) => [`$${Number(val).toLocaleString()}`, 'Avg Sale Price']}
                        labelFormatter={(label) => `Built in ${label}`}
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }}
                      />
                      <Area type="monotone" dataKey="avg_price" stroke="#4f46e5" strokeWidth={3} fillOpacity={1} fill="url(#priceGradient)" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="loading-text">Loading trends chart...</p>
                )}
              </div>
            </div>

            {/* Chart 2: Price vs Size Brackets */}
            <div className="card chart-card">
              <div className="chart-header">
                <div>
                  <p className="section-label">PROPERTY BRACKETS</p>
                  <h3>Average Price by Square Footage Tier</h3>
                </div>
              </div>

              <div className="chart-wrapper">
                {sizeBrackets.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={sizeBrackets} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="size_tier" stroke="#64748b" />
                      <YAxis stroke="#64748b" tickFormatter={(val) => `$${val / 1000}k`} />
                      <Tooltip
                        formatter={(val) => [`$${Number(val).toLocaleString()}`, 'Avg Price']}
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }}
                      />
                      <Bar dataKey="avg_price" fill="#2563eb" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="loading-text">Loading size brackets chart...</p>
                )}
              </div>
            </div>
          </div>

          {/* Feature Importance & Model Weights */}
          <div className="card feature-card">
            <div className="chart-header">
              <div>
                <p className="section-label">XGBOOST EXPLANABILITY</p>
                <h3>Model Feature Weight Distribution</h3>
                <p className="card-subtext">Relative percentage weight assigned to each input feature by the XGBoost decision tree ensemble.</p>
              </div>
            </div>

            <div className="features-list">
              {featureImportance.length > 0 ? (
                featureImportance.map((item, idx) => (
                  <div key={idx} className="feature-item">
                    <div className="feature-info">
                      <span className="feature-name">{item.feature}</span>
                      <span className="feature-val">{item.importance}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div
                        className="progress-bar-fill"
                        style={{ width: `${item.importance}%` }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p className="loading-text">Loading feature weights...</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App