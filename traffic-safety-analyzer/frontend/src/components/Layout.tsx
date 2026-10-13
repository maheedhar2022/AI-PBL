import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Video, AlertTriangle, BarChart2,
  Settings, BookOpen, Shield
} from 'lucide-react'

const navItems = [
  { to: '/overview',    label: 'Overview',    icon: LayoutDashboard },
  { to: '/monitoring',  label: 'Live Monitor',icon: Video },
  { to: '/incidents',   label: 'Incidents',   icon: AlertTriangle },
  { to: '/evaluation',  label: 'AI Evaluation', icon: BarChart2 },
  { to: '/settings',    label: 'Settings',    icon: Settings },
  { to: '/docs',        label: 'Docs',        icon: BookOpen },
]

export default function Layout() {
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
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `sidebar-link${isActive ? ' active' : ''}`
              }
            >
              <Icon className="w-5 h-5 shrink-0" />
              <span className="text-sm">{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 text-xs text-gray-600">
          v1.0.0 · Research Build
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-950">
        <Outlet />
      </main>
    </div>
  )
}

