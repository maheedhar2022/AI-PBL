import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
})

// Dashboard
export const getDashboardStats = () => api.get('/dashboard/stats').then(r => r.data)
export const getRecentIncidents = (limit = 10) =>
  api.get('/dashboard/recent', { params: { limit } }).then(r => r.data)
export const getDashboardAnalytics = () => api.get('/dashboard/analytics').then(r => r.data)

// Camera
export const getCameraStatus = () => api.get('/camera/status').then(r => r.data)
export const configureCamera = (source: string, fps_target = 10) =>
  api.post('/camera/configure', { source, fps_target }).then(r => r.data)
export const uploadVideoFile = (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/camera/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

// Monitoring & Evidence
export const startMonitoring = () => api.post('/monitoring/start').then(r => r.data)
export const stopMonitoring = () => api.post('/monitoring/stop').then(r => r.data)
export const getMonitoringStatus = () => api.get('/monitoring/status').then(r => r.data)
export const getLatestFrame = () => api.get('/monitoring/frame').then(r => r.data)

export const startRecording = () => api.post('/monitoring/record/start').then(r => r.data)
export const stopRecording = () => api.post('/monitoring/record/stop').then(r => r.data)
export const listRecordings = () => api.get('/monitoring/recordings').then(r => r.data)
export const deleteRecording = (filename: string) =>
  api.delete(`/monitoring/recordings/${filename}`).then(r => r.data)

export const takeSnapshot = () => api.post('/monitoring/snapshot').then(r => r.data)
export const listSnapshots = () => api.get('/monitoring/snapshots').then(r => r.data)
export const deleteSnapshot = (filename: string) =>
  api.delete(`/monitoring/snapshots/${filename}`).then(r => r.data)

// Incidents
export interface IncidentFilters {
  violation_type?: string
  review_status?: string
  camera_source?: string
  vehicle_number?: string
  start_time?: string
  end_time?: string
  page?: number
  page_size?: number
}
export const listIncidents = (params: IncidentFilters = {}) =>
  api.get('/incidents/', { params }).then(r => r.data)
export const getIncident = (id: string) => api.get(`/incidents/${id}`).then(r => r.data)
export const reviewIncident = (id: string, status: string, notes?: string) =>
  api.patch(`/incidents/${id}/review`, { status, notes }).then(r => r.data)
export const deleteIncident = (id: string) => api.delete(`/incidents/${id}`)
export const batchReviewIncidents = (incidentIds: string[], status: string, notes?: string) =>
  api.post('/incidents/batch-review', { incident_ids: incidentIds, status, notes }).then(r => r.data)
export const bulkDeleteIncidents = (incidentIds: string[]) =>
  api.post('/incidents/bulk-delete', { incident_ids: incidentIds }).then(r => r.data)

export const exportIncidents = async (params: IncidentFilters & { format?: 'csv' | 'json' }) => {
  const format = params.format || 'csv'
  const response = await api.get('/incidents/export', {
    params,
    responseType: 'blob',
  })
  const blob = new Blob([response.data], {
    type: format === 'json' ? 'application/json' : 'text/csv',
  })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `incidents_export.${format}`
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

// Evaluation
export const runEvaluation = (body: {
  dataset_path: string
  dataset_name: string
  iou_threshold?: number
  confidence_threshold?: number
  experiment_name?: string
  notes?: string
}) => api.post('/evaluation/run', body).then(r => r.data)
export const getEvalStatus = () => api.get('/evaluation/status').then(r => r.data)
export const listEvalResults = () => api.get('/evaluation/results').then(r => r.data)
export const getEvalResult = (id: string) => api.get(`/evaluation/results/${id}`).then(r => r.data)
export const deleteEvalResult = (id: string) => api.delete(`/evaluation/results/${id}`).then(r => r.data)
export const listDatasets = () => api.get('/evaluation/datasets').then(r => r.data)

// Settings
export const getSettings = () => api.get('/settings/').then(r => r.data)
export const updateSettings = (data: Record<string, unknown>) =>
  api.patch('/settings/', data).then(r => r.data)
export const getModelInfo = () => api.get('/settings/model-info').then(r => r.data)

// Deep Learning Models Registry & Benchmark
export const getModelsInfo = () => api.get('/models/info').then(r => r.data)
export const runModelsBenchmark = (iterations: number = 10) =>
  api.post('/models/benchmark', { iterations }).then(r => r.data)
