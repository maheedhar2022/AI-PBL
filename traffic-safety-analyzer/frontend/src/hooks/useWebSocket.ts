import { useEffect, useRef, useCallback } from 'react'

export function useWebSocket(
  onMessage: (data: Record<string, unknown>) => void,
  enabled = true
) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<number | null>(null)

  const connect = useCallback(() => {
    if (!enabled) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}/ws/live`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected')
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        onMessage(data)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      console.log('[WS] Disconnected, reconnecting in 3s…')
      reconnectTimer.current = window.setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }

    // Heartbeat
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping')
    }, 15000)

    return () => {
      clearInterval(ping)
      ws.close()
    }
  }, [enabled, onMessage])

  useEffect(() => {
    const cleanup = connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      cleanup?.()
    }
  }, [connect])
}

