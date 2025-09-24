'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Activity,
  Brain,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader2,
  Zap,
  Eye,
  FileText,
  ChevronDown,
  Terminal,
  Cpu,
  Database
} from 'lucide-react'

interface LogEntry {
  timestamp: string
  level: 'info' | 'warning' | 'error' | 'success'
  service: string
  message: string
  metadata?: Record<string, any>
}

interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  active_processes: number
  requests_per_minute: number
}

export default function MissionControl() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    active_processes: 0,
    requests_per_minute: 0
  })
  const [activeTab, setActiveTab] = useState<'logs' | 'metrics' | 'ai'>('logs')
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [aiAnalysis, setAiAnalysis] = useState('')
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Check Kamiwaza connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001'}/api/health`)
        const data = await response.json()

        if (data.kamiwaza?.healthy) {
          setIsConnected(true)
          setConnectionError(null)
        } else {
          setIsConnected(false)
          setConnectionError(data.kamiwaza?.message || 'Kamiwaza is offline')
        }
      } catch (error) {
        setIsConnected(false)
        setConnectionError('Failed to connect to backend service')
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'info': return 'text-blue-600 bg-blue-100'
      case 'warning': return 'text-yellow-600 bg-yellow-100'
      case 'error': return 'text-red-600 bg-red-100'
      case 'success': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getLevelIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'info': return <AlertCircle className="w-4 h-4" />
      case 'warning': return <AlertCircle className="w-4 h-4" />
      case 'error': return <XCircle className="w-4 h-4" />
      case 'success': return <CheckCircle className="w-4 h-4" />
      default: return <AlertCircle className="w-4 h-4" />
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Mission Control</h3>
              <p className="text-xs opacity-90">Real-time System Monitoring</p>
            </div>
          </div>
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            isConnected ? 'bg-green-400/20 text-green-100' : 'bg-red-400/20 text-red-100'
          }`}>
            {isConnected ? '● Connected' : '○ Disconnected'}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab('logs')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'logs'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50 dark:bg-blue-900/20'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <Terminal className="w-4 h-4 inline mr-2" />
          Logs
        </button>
        <button
          onClick={() => setActiveTab('metrics')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'metrics'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50 dark:bg-blue-900/20'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <Cpu className="w-4 h-4 inline mr-2" />
          Metrics
        </button>
        <button
          onClick={() => setActiveTab('ai')}
          className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'ai'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50 dark:bg-blue-900/20'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
          }`}
        >
          <Brain className="w-4 h-4 inline mr-2" />
          AI Analysis
        </button>
      </div>

      {/* Content */}
      <div className="h-[400px] overflow-hidden">
        {/* Logs Tab */}
        {activeTab === 'logs' && (
          <div className="h-full overflow-y-auto p-4">
            {!isConnected ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <XCircle className="w-12 h-12 text-red-500 mb-4" />
                <p className="text-lg font-medium mb-2">Mission Control Offline</p>
                <p className="text-sm text-center">{connectionError || 'Kamiwaza service is not available'}</p>
              </div>
            ) : logs.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                <p>No logs available. Waiting for system activity...</p>
              </div>
            ) : (
              <div className="space-y-2">
                {logs.map((log, index) => (
                  <div key={index} className="flex items-start gap-3 text-xs font-mono">
                    <span className="text-gray-400 dark:text-gray-500 min-w-[140px]">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                    <span className={`px-2 py-1 rounded-full ${getLevelColor(log.level)}`}>
                      {getLevelIcon(log.level)}
                    </span>
                    <span className="text-purple-600 dark:text-purple-400 min-w-[100px]">
                      [{log.service}]
                    </span>
                    <span className="text-gray-700 dark:text-gray-300 flex-1">
                      {log.message}
                    </span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        )}

        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="p-6">
            {!isConnected ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <Database className="w-12 h-12 text-red-500 mb-4" />
                <p className="text-lg font-medium mb-2">Metrics Unavailable</p>
                <p className="text-sm text-center">Connect to Kamiwaza to view system metrics</p>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">CPU Usage</span>
                  <Cpu className="w-4 h-4 text-gray-400" />
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.cpu_usage.toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500"
                    style={{ width: `${metrics.cpu_usage}%` }}
                  />
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Memory</span>
                  <Database className="w-4 h-4 text-gray-400" />
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.memory_usage.toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-teal-600 transition-all duration-500"
                    style={{ width: `${metrics.memory_usage}%` }}
                  />
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Active Processes</span>
                  <Zap className="w-4 h-4 text-gray-400" />
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.active_processes}
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Requests/min</span>
                  <Activity className="w-4 h-4 text-gray-400" />
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metrics.requests_per_minute}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    )}

        {/* AI Analysis Tab */}
        {activeTab === 'ai' && (
          <div className="p-6">
            {!isConnected ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <Brain className="w-12 h-12 text-red-500 mb-4" />
                <p className="text-lg font-medium mb-2">AI Analysis Unavailable</p>
                <p className="text-sm text-center">Kamiwaza connection required for AI insights</p>
              </div>
            ) : (
              <>
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4 mb-4">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">AI System Status</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Waiting for Kamiwaza models to initialize...
                  </p>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Model Health</span>
                    <span className="text-sm font-medium text-yellow-600">Initializing...</span>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Queue Status</span>
                    <span className="text-sm font-medium text-gray-600">No data</span>
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Optimization</span>
                    <span className="text-sm font-medium text-gray-600">Pending</span>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}