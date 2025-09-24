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
  Eye
} from 'lucide-react'
import axios from 'axios'
import ArchitectureSelector, { ProcessingMode } from '../components/ArchitectureSelector'

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

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
  const [pendingRFPs, setPendingRFPs] = useState<RFP[]>([])
  const [processingStatus, setProcessingStatus] = useState<Record<string, RFPStatus>>({})
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [architectureMode, setArchitectureMode] = useState<ProcessingMode>('classic')
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)

  const fetchPendingRFPs = async () => {
    setRefreshing(true)
    setError(null)
    try {
      const response = await axios.get(`${API_URL}/api/rfps/pending`)
      setPendingRFPs(response.data.rfps)
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
      response.data.rfps.forEach((status: RFPStatus) => {
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
      // For comparison mode, use special endpoint
      const endpoint = architectureMode === 'comparison'
        ? `${API_URL}/api/rfps/process-comparison`
        : `${API_URL}/api/rfps/process`

      const response = await axios.post(endpoint, {
        force_process_all: false,
        architecture_mode: architectureMode,
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
    
    // Poll for status every 5 seconds
    const interval = setInterval(() => {
      fetchAllStatus()
    }, 5000)
    
    return () => clearInterval(interval)
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Mercury RFP Automation</h1>
              <p className="text-gray-600 mt-1">Powered by GPT-5 and Kamiwaza AI</p>
            </div>
            <div className="flex gap-3">
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
                className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
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
        </div>

        {/* Architecture Selector */}
        <ArchitectureSelector
          currentMode={architectureMode}
          onModeChange={setArchitectureMode}
          sessionId={sessionId}
        />

        {/* Pending RFPs */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Pending RFPs ({pendingRFPs.length})
          </h2>
          
          {pendingRFPs.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              No RFPs with checked boxes found in the Google Sheet
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
                          className="text-sm text-primary-600 hover:text-primary-700 mt-1 inline-flex items-center gap-1"
                        >
                          <Eye className="w-3 h-3" />
                          View on SAM.gov
                        </a>
                      )}
                    </div>
                    <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">
                      Ready to Process
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Processing Status */}
        {Object.keys(processingStatus).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Loader2 className="w-5 h-5" />
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
                      className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${status.progress * 100}%` }}
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
      </div>
    </div>
  )
}