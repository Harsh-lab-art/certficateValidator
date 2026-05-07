import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
})

api.interceptors.request.use(cfg => {
  const token = localStorage.getItem('cv_token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

api.interceptors.response.use(
  r => r,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('cv_token')
    }
    return Promise.reject(err)
  }
)

export default api

export const verify = {
  submit:   (formData) => api.post('/verify', formData),
  poll:     (id)       => api.get(`/verify/${id}`),
  heatmap:  (id)       => `/api/v1/verify/${id}/heatmap`,
  report:   (id)       => `/api/v1/verify/${id}/report`,
  history:  (p = {})   => api.get('/verify', { params: p }),
}

export const institutions = {
  search: (q, limit = 10) => api.get('/institutions/search', { params: { q, limit } }),
  lookup: (name)          => api.get('/institutions/lookup', { params: { name } }),
  list:   ()              => api.get('/institutions'),
}

export const stats = {
  get: () => api.get('/stats'),
}

export const auth = {
  login:    (email, password) =>
    api.post('/auth/login', new URLSearchParams({ username: email, password }),
             { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }),
  me:       () => api.get('/auth/me'),
}
