import os
import json
import logging
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = os.getenv("TRACKING_SHEET_ID")
        self.range_name = os.getenv("SHEET_RANGE", "A1:Z1000")
        self.service = self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Sheets API service"""
        try:
            service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
            
            if not service_account_file or not os.path.exists(service_account_file):
                raise FileNotFoundError(f"Service account file not found: {service_account_file}")
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            service = build('sheets', 'v4', credentials=credentials)
            logger.info("Google Sheets service initialized successfully")
            return service
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets service: {e}")
            raise
    
    def get_checked_rfps(self, sheet_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all RFPs with checked boxes in column B"""
        try:
            sheet_id = sheet_id or self.sheet_id

            result = self.service.spreadsheets().values().get(
                spreadsheetId=sheet_id,
                range=self.range_name,
                valueRenderOption='FORMULA'  # Get formulas to extract hyperlinks
            ).execute()

            values = result.get('values', [])

            if not values:
                logger.warning("No data found in sheet")
                return []

            headers = values[0] if values else []
            checked_rfps = []

            checkbox_col_index = 1  # Column B (0-indexed)
            folder_col_index = 12  # Column M (0-indexed)

            for row_idx, row in enumerate(values[1:], start=2):
                if len(row) > checkbox_col_index:
                    checkbox_value = row[checkbox_col_index]

                    if checkbox_value and str(checkbox_value).upper() in ['TRUE', 'YES', 'X', '✓', '1']:
                        rfp_data = {}
                        for col_idx, header in enumerate(headers):
                            if col_idx < len(row):
                                rfp_data[header.lower().replace(' ', '_')] = row[col_idx]

                        rfp_data['row_number'] = row_idx

                        # Extract Google Drive folder URL from column M
                        if len(row) > folder_col_index and row[folder_col_index]:
                            cell_value = row[folder_col_index].strip()

                            # Check if it's a HYPERLINK formula
                            if cell_value.startswith('=HYPERLINK('):
                                # Extract URL from HYPERLINK formula: =HYPERLINK("url", "text")
                                import re
                                match = re.search(r'=HYPERLINK\("([^"]+)"', cell_value)
                                if match:
                                    folder_url = match.group(1)
                                else:
                                    folder_url = cell_value
                            else:
                                folder_url = cell_value

                            if folder_url and ('drive.google.com' in folder_url or folder_url.startswith('http')):
                                rfp_data['drive_folder_url'] = folder_url
                                rfp_data['data_source'] = 'google_drive'
                                logger.info(f"Row {row_idx} has Drive folder: {folder_url[:50]}...")
                            else:
                                rfp_data['data_source'] = 'sam_gov'
                        else:
                            rfp_data['data_source'] = 'sam_gov'

                        if 'notice_id' not in rfp_data:
                            rfp_data['notice_id'] = rfp_data.get('id', f"RFP_{row_idx}")

                        if 'title' not in rfp_data:
                            rfp_data['title'] = rfp_data.get('name', f"RFP {row_idx}")

                        if 'url' not in rfp_data and 'link' in rfp_data:
                            rfp_data['url'] = rfp_data['link']

                        checked_rfps.append(rfp_data)

            logger.info(f"Found {len(checked_rfps)} RFPs with checked boxes")
            drive_count = sum(1 for rfp in checked_rfps if rfp.get('data_source') == 'google_drive')
            logger.info(f"  - {drive_count} with Google Drive folders")
            logger.info(f"  - {len(checked_rfps) - drive_count} with SAM.gov URLs only")
            return checked_rfps
            
        except HttpError as e:
            logger.error(f"Google Sheets API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting checked RFPs: {e}")
            raise
    
    async def update_rfp_status(self, notice_id: str, status: str, folder_url: str = None, error: str = None):
        """Update the status of an RFP in the sheet"""
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.sheet_id,
                range=self.range_name
            ).execute()
            
            values = result.get('values', [])
            headers = values[0] if values else []
            
            notice_id_col = None
            status_col = None
            folder_col = None
            error_col = None
            
            for idx, header in enumerate(headers):
                header_lower = header.lower()
                if 'notice' in header_lower or 'id' in header_lower:
                    notice_id_col = idx
                elif 'status' in header_lower:
                    status_col = idx
                elif 'folder' in header_lower or 'drive' in header_lower:
                    # Skip updating folder column - it should keep original RFP docs
                    # folder_col = idx
                    pass
                elif 'error' in header_lower:
                    error_col = idx
            
            if status_col is None:
                status_col = len(headers)
                headers.append('Status')
            
            if folder_url and folder_col is None:
                folder_col = len(headers)
                headers.append('Drive Folder')
            
            if error and error_col is None:
                error_col = len(headers)
                headers.append('Error')
            
            for row_idx, row in enumerate(values[1:], start=2):
                if notice_id_col is not None and notice_id_col < len(row):
                    if row[notice_id_col] == notice_id:
                        updates = []
                        
                        if status_col is not None:
                            cell_range = f"{chr(65 + status_col)}{row_idx}"
                            updates.append({
                                'range': cell_range,
                                'values': [[status]]
                            })

                        # Update column T with "Latest Submission Run" timestamp
                        if status == 'Processed' or status == 'completed':
                            from datetime import datetime
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            cell_range = f"T{row_idx}"  # Column T for timestamp
                            updates.append({
                                'range': cell_range,
                                'values': [[timestamp]]
                            })
                        
                        # DO NOT update folder column - keep original RFP docs
                        # if folder_url and folder_col is not None:
                        #     cell_range = f"{chr(65 + folder_col)}{row_idx}"
                        #     updates.append({
                        #         'range': cell_range,
                        #         'values': [[folder_url]]
                        #     })
                        
                        if error and error_col is not None:
                            cell_range = f"{chr(65 + error_col)}{row_idx}"
                            updates.append({
                                'range': cell_range,
                                'values': [[error]]
                            })
                        
                        if updates:
                            self.service.spreadsheets().values().batchUpdate(
                                spreadsheetId=self.sheet_id,
                                body={'data': updates, 'valueInputOption': 'USER_ENTERED'}
                            ).execute()
                            
                            logger.info(f"Updated status for RFP {notice_id}")
                        break
            
        except Exception as e:
            logger.error(f"Error updating RFP status: {e}")
            raise
    
    async def clear_checkbox(self, row_number: int):
        """Clear the checkbox after processing"""
        try:
            cell_range = f"B{row_number}"
            
            self.service.spreadsheets().values().update(
                spreadsheetId=self.sheet_id,
                range=cell_range,
                valueInputOption='USER_ENTERED',
                body={'values': [[False]]}
            ).execute()
            
            logger.info(f"Cleared checkbox for row {row_number}")
            
        except Exception as e:
            logger.error(f"Error clearing checkbox: {e}")
            raise