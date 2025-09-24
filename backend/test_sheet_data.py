#!/usr/bin/env python3
"""Check what data we're getting from Google Sheets column M"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from services.google_sheets import GoogleSheetsService

load_dotenv()

async def check_sheet_data():
    sheets = GoogleSheetsService()
    rfps = await sheets.get_checked_rfps()

    for rfp in rfps:
        if rfp.get('row_number') == 250:
            print(f"Row 250 data:")
            print(f"  - data_source: {rfp.get('data_source')}")
            print(f"  - drive_folder_url: {rfp.get('drive_folder_url')}")
            print(f"  - drive_folder (raw): {rfp.get('drive_folder')}")
            print(f"  - All keys: {list(rfp.keys())}")
            break

    return rfps

if __name__ == "__main__":
    asyncio.run(check_sheet_data())