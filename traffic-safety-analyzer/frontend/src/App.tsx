import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import LiveMonitoring from './pages/LiveMonitoring'
import IncidentExplorer from './pages/IncidentExplorer'
import EvaluationPage from './pages/EvaluationPage'
import SettingsPage from './pages/SettingsPage'
import DocumentationPage from './pages/DocumentationPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<Overview />} />
          <Route path="monitoring" element={<LiveMonitoring />} />
          <Route path="incidents" element={<IncidentExplorer />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="docs" element={<DocumentationPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

