import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts'
import {
  Home, TrendingUp, BarChart3, ShieldCheck, Zap, Layers, Activity, Building2,
  Printer, ArrowUpRight, Sparkles, Car, Star
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const initialFormData = {
  Gr_Liv_Area: 1800,
  Bedroom_AbvGr: 3,
  Year_Built: 2005,
  Full_Bath: 2,
  Overall_Qual: 7,
  Garage_Cars: 2
}

const qualityLabels = {
  1: 'Very Poor',
  2: 'Poor',
  3: 'Fair',
  4: 'Below Average',
  5: 'Average',
  6: 'Above Average',
  7: 'Good',
  8: 'Very Good',
  9: 'Excellent',
  10: 'Luxury / Custom'
}

function App() {
  const [activeTab, setActiveTab] = useState('valuation') // 'valuation' | 'analytics'
  const [formData, setFormData] = useState(initialFormData)
  const [prediction, setPrediction] = useState(null)
  const [predictionDetails, setPredictionDetails] = useState(null)
  const [roiUpgrades, setRoiUpgrades] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Analytics State
  const [kpis, setKpis] = useState(null)
  const [trends, setTrends] = useState([])
  const [sizeBrackets, setSizeBrackets] = useState([])
  const [neighborhoods, setNeighborhoods] = useState([])
  const [featureImportance, setFeatureImportance] = useState([])

  useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    try {
      const [kpiRes, trendsRes, bracketsRes, featuresRes, neighRes] = await Promise.all([
        axios.get(`${API_BASE}/analytics/kpis`),
        axios.get(`${API_BASE}/analytics/trends`),
        axios.get(`${API_BASE}/analytics/price-vs-sqft`),
        axios.get(`${API_BASE}/analytics/feature-importance`),
        axios.get(`${API_BASE}/analytics/neighborhoods`)
      ])
      setKpis(kpiRes.data)
      setTrends(trendsRes.data)
      setSizeBrackets(bracketsRes.data)
      setFeatureImportance(featuresRes.data)
      setNeighborhoods(neighRes.data)
    } catch (err) {
      console.warn('Analytics API initializing or unavailable:', err.message)
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
    setRoiUpgrades([])

    try {
      const [predRes, roiRes] = await Promise.all([
        axios.post(`${API_BASE}/predict`, formData),
        axios.post(`${API_BASE}/analytics/roi-simulator`, formData)
      ])

      const price = predRes.data.predicted_price
      const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
      }).format(price)

      setPrediction(formattedPrice)
      setPredictionDetails({
        rawPrice: price,
        pricePerSqFt: Math.round(predRes.data.price_per_sqft || (price / formData.Gr_Liv_Area)),
        percentile: predRes.data.market_percentile || 50
      })
      setRoiUpgrades(roiRes.data.upgrades || [])
    } catch (err) {
      console.error('Error fetching prediction:', err)
      setError('Unable to connect to the valuation service. Please ensure the API server is running at http://127.0.0.1:8000.')
    } finally {
      setLoading(false)
    }
  }

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="app-shell">
      {/* Header Navigation */}
      <header className="app-header no-print">
        <div className="header-container">
          <div className="brand">
            <Building2 className="brand-icon" />
            <div className="brand-text">
              <span className="brand-name">RealEstate ML</span>
              <span className="brand-tag">XGBoost Valuation & Market Insights</span>
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
        <section className="kpi-banner no-print">
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
          <section className="hero-panel no-print">
            <div className="hero-copy">
              <p className="eyebrow">HIGH-ACCURACY ML VALUATION</p>
              <h1>Estimate your home's true value in seconds.</h1>
              <p className="hero-text">
                Enter your property details below. Our XGBoost model applies monotonic constraints to guarantee logical valuation scaling based on quality, living area, bathrooms, and garage capacity.
              </p>
              <div className="hero-badges">
                <span><Zap size={14} /> Monotonic XGBoost</span>
                <span><ShieldCheck size={14} /> 88.1% $R^2$ Accuracy</span>
                <span><Sparkles size={14} /> Renovation ROI Simulator</span>
              </div>
            </div>

            <div className="hero-card">
              <h2>Property Snapshot</h2>
              <div className="stats-grid">
                <div>
                  <strong>Living Area</strong>
                  <span>{formData.Gr_Liv_Area || 0} sq ft</span>
                </div>
                <div>
                  <strong>Overall Quality</strong>
                  <span>{formData.Overall_Qual} / 10 ({qualityLabels[formData.Overall_Qual]})</span>
                </div>
                <div>
                  <strong>Bedrooms / Baths</strong>
                  <span>{formData.Bedroom_AbvGr} Bed | {formData.Full_Bath} Bath</span>
                </div>
                <div>
                  <strong>Garage Capacity</strong>
                  <span>{formData.Garage_Cars} Car{formData.Garage_Cars === 1 ? '' : 's'}</span>
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
                    min="300"
                    max="10000"
                    value={formData.Gr_Liv_Area}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label htmlFor="Overall_Qual" className="field">
                  <span>Overall Quality (1 = Poor, 10 = Luxury)</span>
                  <div className="quality-input-wrapper">
                    <input
                      id="Overall_Qual"
                      type="range"
                      name="Overall_Qual"
                      min="1"
                      max="10"
                      value={formData.Overall_Qual}
                      onChange={handleChange}
                      className="range-slider"
                    />
                    <span className="quality-badge">{formData.Overall_Qual} - {qualityLabels[formData.Overall_Qual]}</span>
                  </div>
                </label>

                <label htmlFor="Full_Bath" className="field">
                  <span>Full Bathrooms</span>
                  <input
                    id="Full_Bath"
                    type="number"
                    name="Full_Bath"
                    min="0"
                    max="6"
                    value={formData.Full_Bath}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label htmlFor="Bedroom_AbvGr" className="field">
                  <span>Number of Bedrooms</span>
                  <input
                    id="Bedroom_AbvGr"
                    type="number"
                    name="Bedroom_AbvGr"
                    min="0"
                    max="8"
                    value={formData.Bedroom_AbvGr}
                    onChange={handleChange}
                    required
                  />
                </label>

                <label htmlFor="Garage_Cars" className="field">
                  <span>Garage Capacity (Cars)</span>
                  <input
                    id="Garage_Cars"
                    type="number"
                    name="Garage_Cars"
                    min="0"
                    max="5"
                    value={formData.Garage_Cars}
                    onChange={handleChange}
                    required
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
                    required
                  />
                </label>
              </div>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Evaluating Model Parameters...' : 'Predict Property Value'}
              </button>
            </form>

            <aside className="card result-card">
              <div className="card-header print-header">
                <div>
                  <p className="section-label">SMART ESTIMATE</p>
                  <h2>See your projected value</h2>
                </div>
                {prediction && (
                  <button type="button" onClick={handlePrint} className="print-btn no-print" title="Export PDF / Print Report">
                    <Printer size={16} /> Print Report
                  </button>
                )}
              </div>

              <div className="result-body">
                {loading ? (
                  <div className="result-loading">
                    <div className="spinner" />
                    <p>Evaluating XGBoost model monotonicity constraints...</p>
                  </div>
                ) : prediction ? (
                  <div className="result-success">
                    <span className="result-tag">Estimated Market Valuation</span>
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
                      This estimate is generated by our monotonicity-constrained XGBoost ML model trained on 2,930 Ames housing records.
                    </p>
                  </div>
                ) : error ? (
                  <p className="result-error">{error}</p>
                ) : (
                  <p className="result-placeholder">
                    Adjust property parameters and click <strong>Predict Property Value</strong> to calculate the valuation.
                  </p>
                )}
              </div>

              {/* Renovation ROI Simulator */}
              {roiUpgrades.length > 0 && (
                <div className="roi-section">
                  <div className="roi-header">
                    <Sparkles className="roi-icon" />
                    <h4>Renovation & Upgrade ROI Simulator</h4>
                  </div>
                  <div className="roi-grid">
                    {roiUpgrades.map((item, idx) => (
                      <div key={idx} className="roi-item">
                        <div className="roi-info">
                          <span className="roi-name">{item.name}</span>
                          <span className="roi-value">+${item.added_value?.toLocaleString()}</span>
                        </div>
                        <span className="roi-subtext">New Value: ${item.new_price?.toLocaleString()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </aside>
          </section>
        </div>
      )}

      {/* TAB 2: MARKET ANALYTICS DASHBOARD */}
      {activeTab === 'analytics' && (
        <div className="tab-content analytics-tab">
          <div className="analytics-header">
            <h2>Ames Housing Market Analytics & Neighborhood Insights</h2>
            <p>Interactive data visualizations computed from 2,930 historical real estate transactions.</p>
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
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={trends} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="decade" stroke="#64748b" />
                      <YAxis stroke="#64748b" tickFormatter={(val) => `$${val / 1000}k`} />
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
                  <ResponsiveContainer width="100%" height={280}>
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

          {/* Chart 3: Neighborhood Price Comparison */}
          {neighborhoods.length > 0 && (
            <div className="card chart-card neighborhood-card">
              <div className="chart-header">
                <div>
                  <p className="section-label">LOCATION ANALYSIS</p>
                  <h3>Average Property Price by Neighborhood</h3>
                  <p className="card-subtext">Comparing valuation averages across top Ames, Iowa residential neighborhoods.</p>
                </div>
              </div>

              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={neighborhoods} layout="vertical" margin={{ top: 10, right: 30, left: 70, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis type="number" stroke="#64748b" tickFormatter={(val) => `$${val / 1000}k`} />
                    <YAxis type="category" dataKey="neighborhood" stroke="#64748b" width={80} />
                    <Tooltip
                      formatter={(val) => [`$${Number(val).toLocaleString()}`, 'Avg Price']}
                      contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }}
                    />
                    <Bar dataKey="avg_price" fill="#6366f1" radius={[0, 8, 8, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Feature Importance & Model Weights */}
          <div className="card feature-card">
            <div className="chart-header">
              <div>
                <p className="section-label">XGBOOST EXPLANABILITY</p>
                <h3>Model Feature Weight Distribution</h3>
                <p className="card-subtext">Relative percentage weight assigned to each input parameter by the monotonicity-constrained XGBoost model.</p>
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