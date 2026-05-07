import { useState, useEffect } from 'react'
import { stats } from '../utils/api'

export function useStats(pollInterval = 10000) {
  const [data, setData] = useState({
    total_verifications: 0,
    genuine: 0,
    fake: 0,
    suspicious: 0,
    avg_trust_score: 0,
    fake_rate: 0,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const { data: res } = await stats.get()
        setData(res)
      } catch { /* backend offline — keep zeros */ } finally {
        setLoading(false)
      }
    }
    fetch()
    const id = setInterval(fetch, pollInterval)
    return () => clearInterval(id)
  }, [pollInterval])

  return { data, loading }
}
