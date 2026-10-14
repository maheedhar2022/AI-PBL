import { useEffect, useState } from 'react'
import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Video, AlertTriangle, BarChart2,
  Settings, BookOpen, Shield
} from 'lucide-react'
import { getDashboardStats } from '../services/api'

export default function Layout() {
  const [cameraRunning, setCameraRunning] = useState(false)
  const [pendingCount, setPendingCount] = useState(0)

  useEffect(() => {
    let mounted = true
    const checkStatus = async () => {
      try {
        const stats = await getDashboardStats()
        if (mounted) {
          setCameraRunning(Boolean(stats.camera_running))
          setPendingCount(stats.pending_review || 0)
        }
      } catch {
        // server might be warming up
      }
    }

    checkStatus()
    const timer = setInterval(checkStatus, 8000)
    return () => {
      mounted = false
      clearInterval(timer)
    }
  }, [])

  const navItems = [
    { to: '/overview',   label: 'Overview',      icon: LayoutDashboard },
    {
      to: '/monitoring',
      label: 'Live Monitor',
      icon: Video,
      badge: cameraRunning ? (
        <span className="flex items-center gap-1 text-[10px] font-semibold text-green-400 bg-green-950/80 px-1.5 py-0.5 rounded border border-green-800">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-ping inline-block" />
          LIVE
        </span>
      ) : null,
    },
    {
      to: '/incidents',
      label: 'Incidents',
      icon: AlertTriangle,
      badge: pendingCount > 0 ? (
        <span className="text-[10px] font-bold text-amber-300 bg-amber-950/80 px-1.5 py-0.2 rounded border border-amber-800">
          {pendingCount}
        </span>
      ) : null,
    },
    { to: '/evaluation', label: 'AI Evaluation', icon: BarChart2 },
    { to: '/settings',   label: 'Settings',      icon: Settings },
    { to: '/docs',       label: 'Docs',          icon: BookOpen },
  ]

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-3 p-5 border-b border-gray-800">
          <Shield className="text-blue-400 w-7 h-7" />
          <div>
            <div className="font-bold text-sm text-white leading-tight">Traffic Safety</div>
            <div className="text-xs text-gray-500">AI Analyzer</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {navItems.map(({ to, label, icon: Icon, badge }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `sidebar-link flex items-center justify-between${isActive ? ' active' : ''}`
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="w-5 h-5 shrink-0" />
                <span className="text-sm">{label}</span>
              </div>
              {badge}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 text-xs text-gray-500 flex items-center justify-between">
          <span>v1.0.0 · Research</span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
            Active
          </span>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-950">
        <Outlet />
      </main>
    </div>
  )
}

