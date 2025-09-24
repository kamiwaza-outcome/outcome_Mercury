'use client'

import { useState, useEffect } from 'react'
import {
  Cpu,
  Network,
  Zap,
  BarChart3,
  FlaskConical,
  Settings,
  Info,
  Check
} from 'lucide-react'
import axios from 'axios'

export type ProcessingMode = 'classic' | 'dynamic' | 'comparison' | 'ab_test'

interface ArchitectureSelectorProps {
  onModeChange: (mode: ProcessingMode) => void
  currentMode: ProcessingMode
  sessionId: string
}

interface ArchitectureMetrics {
  total_processed: number
  classic_count: number
  dynamic_count: number
  classic_avg_time: number
  dynamic_avg_time: number
  classic_avg_quality: number
  dynamic_avg_quality: number
}

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function ArchitectureSelector({
  onModeChange,
  currentMode,
  sessionId
}: ArchitectureSelectorProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [metrics, setMetrics] = useState<ArchitectureMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [savedMode, setSavedMode] = useState<ProcessingMode>(currentMode)

  useEffect(() => {
    fetchMetrics()
    fetchCurrentConfig()
  }, [sessionId])

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/architecture/metrics`)
      setMetrics(response.data)
    } catch (err) {
      console.error('Failed to fetch architecture metrics:', err)
    }
  }

  const fetchCurrentConfig = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/architecture/config/${sessionId}`)
      setSavedMode(response.data.mode)
      onModeChange(response.data.mode)
    } catch (err) {
      console.error('Failed to fetch current config:', err)
    }
  }

  const handleModeChange = async (mode: ProcessingMode) => {
    setLoading(true)
    try {
      await axios.post(`${API_URL}/api/architecture/config`, {
        session_id: sessionId,
        mode: mode,
        user_preference: true
      })
      setSavedMode(mode)
      onModeChange(mode)
    } catch (err) {
      console.error('Failed to update architecture mode:', err)
    } finally {
      setLoading(false)
    }
  }

  const architectures = [
    {
      id: 'classic' as ProcessingMode,
      name: 'Classic',
      description: 'Original monolithic orchestration agent',
      icon: <Cpu className="w-5 h-5" />,
      features: [
        'Single orchestration agent',
        'Sequential processing',
        'Proven stability',
        'Consistent output format'
      ],
      color: 'blue'
    },
    {
      id: 'dynamic' as ProcessingMode,
      name: 'Dynamic',
      description: 'Multi-agent system with specialized agents',
      icon: <Network className="w-5 h-5" />,
      features: [
        '17+ specialized agents',
        'Parallel processing',
        'Anti-template engine',
        'Adaptive learning'
      ],
      color: 'purple'
    },
    {
      id: 'comparison' as ProcessingMode,
      name: 'Comparison',
      description: 'Run both architectures for side-by-side comparison',
      icon: <BarChart3 className="w-5 h-5" />,
      features: [
        'Both architectures',
        'Performance metrics',
        'Quality comparison',
        'Best of both worlds'
      ],
      color: 'green'
    },
    {
      id: 'ab_test' as ProcessingMode,
      name: 'A/B Test',
      description: 'Randomly assign architecture for testing',
      icon: <FlaskConical className="w-5 h-5" />,
      features: [
        'Random assignment',
        'Statistical analysis',
        'Performance tracking',
        'Data-driven insights'
      ],
      color: 'orange'
    }
  ]

  const getColorClasses = (color: string, isSelected: boolean) => {
    const colors = {
      blue: isSelected
        ? 'bg-blue-100 border-blue-500 text-blue-900'
        : 'bg-white border-gray-200 hover:border-blue-300',
      purple: isSelected
        ? 'bg-purple-100 border-purple-500 text-purple-900'
        : 'bg-white border-gray-200 hover:border-purple-300',
      green: isSelected
        ? 'bg-green-100 border-green-500 text-green-900'
        : 'bg-white border-gray-200 hover:border-green-300',
      orange: isSelected
        ? 'bg-orange-100 border-orange-500 text-orange-900'
        : 'bg-white border-gray-200 hover:border-orange-300'
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-gray-600" />
          <h2 className="text-xl font-semibold text-gray-900">Processing Architecture</h2>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1"
        >
          <Info className="w-4 h-4" />
          {showDetails ? 'Hide' : 'Show'} Details
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {architectures.map((arch) => (
          <button
            key={arch.id}
            onClick={() => handleModeChange(arch.id)}
            disabled={loading}
            className={`
              relative p-4 rounded-lg border-2 transition-all duration-200
              ${getColorClasses(arch.color, currentMode === arch.id)}
              ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {currentMode === arch.id && (
              <div className="absolute top-2 right-2">
                <Check className="w-5 h-5" />
              </div>
            )}

            <div className="flex items-center gap-2 mb-2">
              {arch.icon}
              <h3 className="font-semibold">{arch.name}</h3>
            </div>

            <p className="text-sm text-gray-600 mb-3">{arch.description}</p>

            {showDetails && (
              <ul className="space-y-1 text-left">
                {arch.features.map((feature, idx) => (
                  <li key={idx} className="text-xs text-gray-500 flex items-start gap-1">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            )}
          </button>
        ))}
      </div>

      {metrics && showDetails && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Performance Metrics</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Total Processed</p>
              <p className="text-lg font-semibold text-gray-900">{metrics.total_processed}</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Classic Avg Time</p>
              <p className="text-lg font-semibold text-blue-900">
                {metrics.classic_avg_time.toFixed(1)}s
              </p>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Dynamic Avg Time</p>
              <p className="text-lg font-semibold text-purple-900">
                {metrics.dynamic_avg_time.toFixed(1)}s
              </p>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Quality Improvement</p>
              <p className="text-lg font-semibold text-green-900">
                {((metrics.dynamic_avg_quality - metrics.classic_avg_quality) / metrics.classic_avg_quality * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {savedMode !== currentMode && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            Mode changed from <strong>{savedMode}</strong> to <strong>{currentMode}</strong>.
            This will be applied to the next RFP processing.
          </p>
        </div>
      )}
    </div>
  )
}