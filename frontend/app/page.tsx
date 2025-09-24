'use client'

import { useState, useEffect } from 'react'
import {
  FileText,
  CheckCircle,
  AlertCircle,
  Clock,
  Play,
  RefreshCw,
  Loader2,
  FolderOpen,
  Download,
  Eye,
  Cpu,
  Settings,
  Activity,
  MessageSquare,
  BarChart3,
  Home as HomeIcon
} from 'lucide-react'
import axios from 'axios'
import AIAssistant from '@/components/AIAssistant'
import MissionControl from '@/components/MissionControl'

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

interface Model {
  id?: string
  name: string
  status: string
  endpoint?: string
}

interface RFP {
  notice_id: string
  title: string
  url?: string
  status?: string
  row_number?: number
}

interface RFPStatus {
  notice_id: string
  title: string
  status: string
  progress: number
  message: string
  documents_generated: string[]
  errors: string[]
}

export default function Home() {
  const [activeView, setActiveView] = useState<'dashboard' | 'assistant' | 'mission'>('dashboard')
  const [pendingRFPs, setPendingRFPs] = useState<RFP[]>([])
  const [processingStatus, setProcessingStatus] = useState<Record<string, RFPStatus>>({})
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Model selection state
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [defaultModel, setDefaultModel] = useState<string>('')
  const [modelLoading, setModelLoading] = useState(false)
  const [kamiwazaStatus, setKamiwazaStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')

  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)

  // Fetch available models from Kamiwaza
  const fetchModels = async () => {
    setModelLoading(true)
    try {
      const response = await axios.get(`${API_URL}/api/models`)
      const modelList = response.data.models || []
      setModels(modelList)
      setDefaultModel(response.data.default_model || 'llama3')

      if (modelList.length > 0) {
        setSelectedModel(modelList[0].name)
        setKamiwazaStatus('connected')
      } else {
        // Use default model if no models available
        setSelectedModel(response.data.default_model || 'llama3')
        setKamiwazaStatus('disconnected')
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
      setKamiwazaStatus('disconnected')
      // Set default model even on error
      setSelectedModel('llama3')
    } finally {
      setModelLoading(false)
    }
  }

  // Check Kamiwaza health status
  const checkKamiwazaHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`)
      const isHealthy = response.data.kamiwaza?.healthy || false
      setKamiwazaStatus(isHealthy ? 'connected' : 'disconnected')
    } catch (err) {
      setKamiwazaStatus('disconnected')
    }
  }

  // Select a model
  const handleModelChange = async (modelName: string) => {
    setSelectedModel(modelName)
    try {
      await axios.post(`${API_URL}/api/models/select`, { model_name: modelName })
    } catch (err) {
      console.error('Failed to select model:', err)
    }
  }

  const fetchPendingRFPs = async () => {
    setRefreshing(true)
    setError(null)
    try {
      const response = await axios.get(`${API_URL}/api/rfps/pending`)
      setPendingRFPs(response.data.rfps || [])
    } catch (err) {
      setError('Failed to fetch pending RFPs')
      console.error(err)
    } finally {
      setRefreshing(false)
    }
  }

  const fetchAllStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/rfps/status`)
      const statusMap: Record<string, RFPStatus> = {}
      response.data.rfps?.forEach((status: RFPStatus) => {
        statusMap[status.notice_id] = status
      })
      setProcessingStatus(statusMap)
    } catch (err) {
      console.error('Failed to fetch status:', err)
    }
  }

  const processRFPs = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post(`${API_URL}/api/rfps/process`, {
        force_process_all: false,
        model_name: selectedModel,
        session_id: sessionId
      })

      if (response.data.rfps && response.data.rfps.length > 0) {
        // Start polling for status
        const pollInterval = setInterval(() => {
          fetchAllStatus()
        }, 2000)

        // Stop polling after 10 minutes
        setTimeout(() => clearInterval(pollInterval), 600000)
      }

      // Refresh the pending list
      await fetchPendingRFPs()
    } catch (err) {
      setError('Failed to process RFPs')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPendingRFPs()
    fetchAllStatus()
    fetchModels()
    checkKamiwazaHealth()

    // Poll for status every 5 seconds
    const interval = setInterval(() => {
      fetchAllStatus()
    }, 5000)

    // Check Kamiwaza health every 30 seconds
    const healthInterval = setInterval(() => {
      checkKamiwazaHealth()
    }, 30000)

    return () => {
      clearInterval(interval)
      clearInterval(healthInterval)
    }
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />
      case 'in_progress':
      case 'downloading':
      case 'analyzing':
      case 'generating':
      case 'reviewing':
      case 'uploading':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      case 'queued':
        return <Clock className="w-5 h-5 text-gray-500" />
      default:
        return <FileText className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'in_progress':
      case 'downloading':
      case 'analyzing':
      case 'generating':
      case 'reviewing':
      case 'uploading':
        return 'bg-blue-100 text-blue-800'
      case 'queued':
        return 'bg-gray-100 text-gray-800'
      default:
        return 'bg-gray-50 text-gray-600'
    }
  }

  const getKamiwazaStatusColor = () => {
    switch (kamiwazaStatus) {
      case 'connected':
        return 'text-green-600 bg-green-100'
      case 'disconnected':
        return 'text-red-600 bg-red-100'
      case 'checking':
        return 'text-yellow-600 bg-yellow-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header with Navigation */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Mercury RFP Automation</h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Powered by Kamiwaza Local AI Models</p>
            </div>
            <div className="flex gap-3 items-center">
              {/* Kamiwaza Status */}
              <div className={`px-3 py-2 rounded-lg flex items-center gap-2 ${getKamiwazaStatusColor()}`}>
                <Cpu className="w-4 h-4" />
                <span className="text-sm font-medium">
                  {kamiwazaStatus === 'checking' ? 'Checking...' :
                   kamiwazaStatus === 'connected' ? 'Kamiwaza Connected' : 'Kamiwaza Offline'}
                </span>
              </div>

              <button
                onClick={fetchPendingRFPs}
                disabled={refreshing}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <button
                onClick={processRFPs}
                disabled={loading || pendingRFPs.length === 0}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Process RFPs
                  </>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {/* Navigation Tabs */}
          <div className="flex gap-2 mt-6 border-t pt-4">
            <button
              onClick={() => setActiveView('dashboard')}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                activeView === 'dashboard'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <HomeIcon className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={() => setActiveView('assistant')}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                activeView === 'assistant'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              AI Assistant
            </button>
            <button
              onClick={() => setActiveView('mission')}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                activeView === 'mission'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Mission Control
            </button>
          </div>
        </div>

        {/* AI Assistant View */}
        {activeView === 'assistant' && (
          <div className="mb-8">
            <AIAssistant />
          </div>
        )}

        {/* Mission Control View */}
        {activeView === 'mission' && (
          <div className="mb-8">
            <MissionControl />
          </div>
        )}

        {/* Dashboard View */}
        {activeView === 'dashboard' && (
          <>
            {/* Model Selection Panel */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              AI Model Configuration
            </h2>
            <button
              onClick={fetchModels}
              disabled={modelLoading}
              className="px-3 py-1 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2 text-sm"
            >
              <RefreshCw className={`w-3 h-3 ${modelLoading ? 'animate-spin' : ''}`} />
              Refresh Models
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select AI Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={modelLoading}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {models.length > 0 ? (
                  models.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name} {model.status === 'available' && '✓'}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="llama3">Llama 3 (Default)</option>
                    <option value="mistral">Mistral (Fallback)</option>
                    <option value="gpt-4">GPT-4</option>
                  </>
                )}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {models.length > 0 ? `${models.length} models available` : 'Using mock models (Kamiwaza not connected)'}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-xs text-gray-500">Current Model</p>
                <p className="text-sm font-medium text-gray-900">{selectedModel || 'None'}</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-xs text-gray-500">Default Model</p>
                <p className="text-sm font-medium text-gray-900">{defaultModel || 'llama3'}</p>
              </div>
            </div>
          </div>

          {modelLoading && (
            <div className="mt-3 flex items-center gap-2 text-sm text-blue-600">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading available models...
            </div>
          )}
        </div>

            {/* Pending RFPs */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Pending RFPs ({pendingRFPs.length})
          </h2>

          {pendingRFPs.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No RFPs found. Click Refresh to check for new RFPs.
            </p>
          ) : (
            <div className="space-y-3">
              {pendingRFPs.map((rfp) => (
                <div key={rfp.notice_id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900">{rfp.title}</h3>
                      <p className="text-sm text-gray-500 mt-1">Notice ID: {rfp.notice_id}</p>
                      {rfp.url && (
                        <a
                          href={rfp.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-blue-600 hover:text-blue-700 mt-1 inline-flex items-center gap-1"
                        >
                          <Eye className="w-3 h-3" />
                          View on SAM.gov
                        </a>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">
                        Ready to Process
                      </span>
                      {selectedModel && (
                        <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                          {selectedModel}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

            {/* Processing Status */}
            {Object.keys(processingStatus).length > 0 && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Processing Status
            </h2>

            <div className="space-y-4">
              {Object.values(processingStatus).map((status) => (
                <div key={status.notice_id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(status.status)}
                        <h3 className="font-medium text-gray-900">{status.title}</h3>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">Notice ID: {status.notice_id}</p>
                    </div>
                    <span className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(status.status)}`}>
                      {status.status.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${status.progress}%` }}
                    />
                  </div>

                  <p className="text-sm text-gray-600">{status.message}</p>

                  {/* Generated Documents */}
                  {status.documents_generated.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <p className="text-sm font-medium text-gray-700 mb-2">Generated Documents:</p>
                      <div className="flex flex-wrap gap-2">
                        {status.documents_generated.map((doc, idx) => (
                          <span key={idx} className="px-2 py-1 bg-green-50 text-green-700 text-xs rounded">
                            {doc}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Errors */}
                  {status.errors.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <p className="text-sm font-medium text-red-700 mb-2">Errors:</p>
                      {status.errors.map((error, idx) => (
                        <p key={idx} className="text-sm text-red-600">{error}</p>
                      ))}
                    </div>
                  )}

                  {/* View in Drive button for completed */}
                  {status.status === 'completed' && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <button className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2 text-sm">
                        <FolderOpen className="w-4 h-4" />
                        View in Google Drive
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}