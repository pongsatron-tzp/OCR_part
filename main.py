import os
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
import datetime
import re

from PIL import Image
import numpy as np
import cv2

try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False

import google.generativeai as genai
import json
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY and _dotenv_available:
    try:
        dotenv_path_key = os.path.join(os.path.dirname(__file__), 'key.env')
        if os.path.exists(dotenv_path_key):
            load_dotenv(dotenv_path=dotenv_path_key)
            logging.info("โหลดตัวแปรสภาพแวดล้อมจาก key.env แล้ว")
        else:
            load_dotenv()
            logging.info("โหลดตัวแปรสภาพแวดล้อมจาก .env หรือจากสภาพแวดล้อมของระบบแล้ว")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    except Exception as e:
        logging.warning(f"ข้อผิดพลาดในการโหลดตัวแปรสภาพแวดล้อมจากไฟล์: {e}")

# โฟลเดอร์ชั่วคราวสำหรับเก็บภาพที่ดาวน์โหลดจาก Google Drive
TEMP_DOWNLOAD_DIR = "temp_downloaded_images"
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

OUTPUT_MARKDOWN_FILE = "document_ocr_results.md"
CONCURRENCY_OCR = 5

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.readonly' # เปลี่ยนเป็น readonly เพื่อความปลอดภัย
]
SPREADSHEET_ID = '1_6i1xyBLKiUihb-sEcE9AhEyYAL0h-tRtLdX4_lWWKc'
SHEET_NAME = 'Part_OCR'

# ขนาดมาตรฐานของ A4 ที่ 600 DPI (พิกเซล)
A4_WIDTH_PX = 4961
A4_HEIGHT_PX = 7016

async def preprocess_image_for_a4(image_path: str) -> Image.Image:
    """
    ปรับขนาดและเพิ่มขอบขาว (Padding) ให้รูปภาพมีขนาด A4 (4961x7016 พิกเซล)
    โดยรักษาสัดส่วนเดิมของภาพไว้ตรงกลาง
    """
    pil_image = await asyncio.to_thread(Image.open, image_path)
    original_width, original_height = pil_image.size

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    a4_aspect_ratio = A4_WIDTH_PX / A4_HEIGHT_PX
    original_aspect_ratio = original_width / original_height

    new_width, new_height = original_width, original_height

    if original_aspect_ratio > a4_aspect_ratio:
        new_width = A4_WIDTH_PX
        new_height = int(new_width / original_aspect_ratio)
    else:
        new_height = A4_HEIGHT_PX
        new_width = int(new_height * original_aspect_ratio)

    interpolation_method = cv2.INTER_AREA
    if new_width > original_width or new_height > original_height:
        interpolation_method = cv2.INTER_CUBIC

    resized_cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=interpolation_method)
    resized_pil_image = Image.fromarray(cv2.cvtColor(resized_cv_image, cv2.COLOR_BGR2RGB))

    # สร้างภาพพื้นหลังสีขาวขนาด A4
    padded_image = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), color='white')

    # คำนวณตำแหน่งที่จะวางภาพที่ปรับขนาดแล้วให้อยู่ตรงกลาง
    x_offset = (A4_WIDTH_PX - new_width) // 2
    y_offset = (A4_HEIGHT_PX - new_height) // 2

    # วางภาพที่ปรับขนาดแล้วลงบนพื้นหลังสีขาว
    padded_image.paste(resized_pil_image, (x_offset, y_offset))

    logging.info(f"ปรับขนาดภาพ {os.path.basename(image_path)} เป็น {resized_pil_image.size} และเพิ่มขอบขาว (padding) ให้เป็นขนาด A4 ({A4_WIDTH_PX}x{A4_HEIGHT_PX})")

    return padded_image

async def async_ocr_document_from_image(
    image_path: str,
    model: genai.GenerativeModel,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, List[Dict[str, str]] | None, Dict[str, int] | None]:
    image_filename = os.path.basename(image_path)
    if model is None:
        logging.error(f"async_ocr_document_from_image ({image_filename}): โมเดล Gemini เป็น None ข้ามการประมวลผล")
        return image_path, None, None

    if not os.path.exists(image_path):
        logging.error(f"async_ocr_document_from_image ({image_filename}): ไม่พบไฟล์รูปภาพที่ {image_path}")
        return image_path, None, None

    async with semaphore:
        logging.info(f"กำลังเริ่มประมวลผล OCR เอกสารสำหรับรูปภาพ: {image_filename}")
        extracted_data: List[Dict[str, str]] | None = None
        usage_metadata: Dict[str, int] | None = None

        try:
            pil_image = await preprocess_image_for_a4(image_path)
            
            if pil_image is None:
                logging.error(f"async_ocr_document_from_image ({image_filename}): ไม่สามารถอ่านหรือประมวลผลไฟล์รูปภาพได้: {image_path}")
                return image_path, None, None
            
            logging.info(f"         กำลังเรียก Gemini API สำหรับรูปภาพ {image_filename} เพื่อ OCR เอกสาร...")

            prompt_text = (
                "จากภาพที่ให้มา เป็นเอกสารเกี่ยวกับอะไหล่รถยนต์ ซึ่ง**เป็นเอกสารขนาด A4 เสมอ** "
                "โปรดทำการ OCR และดึงข้อมูลที่มีความเกี่ยวข้องกับเบอร์อะไหล่รถยนต์ทั้งหมด "
                "รวมถึงชื่ออะไหล่ รุ่นรถ ปีรถ และราคาของอะไหล่นั้นด้วย "
                "ส่วนข้อมูลที่ไม่เกี่ยวข้องกับเบอร์อะไหล่ ไม่ต้องแสดงผล "
                "ให้ตอบกลับมาเป็น JSON array ของ object โดยที่แต่ละ object "
                "แสดงข้อมูลของอะไหล่แต่ละรายการ "
                "แต่ละ object ต้องมีฟิลด์ดังนี้: 'เบอร์อะไหล่', 'ชื่ออะไหล่', 'รุ่นรถ', 'ปีรถ', 'ราคาปลีก', 'ราคาส่ง', 'Time_stamp' (ใส่เวลาปัจจุบันในรูปแบบ 'YYYY-MM-DD HH:MM:SS') "
                "ถ้าหากไม่มีข้อมูลสำหรับฟิลด์ใดๆ ให้ใส่เป็น 'N/A' "
                "ตัวอย่าง JSON response: "
                "[\n"
                "   {\n"
                "     \"เบอร์อะไหล่\": \"เบอร์อะไหล่ 1\",\n"
                "     \"ชื่ออะไหล่\": \"ชื่ออะไหล่ 1\",\n"
                "     \"รุ่นรถ\": \"รุ่นรถ 1\",\n"
                "     \"ปีรถ\": \"ปีรถ 1\",\n"
                "     \"ราคาปลีก\": \"ราคาปลีก 1\",\n"
                "     \"ราคาส่ง\": \"ราคาส่ง 1\",\n"
                "     \"Time_stamp\": \"2023-10-01 12:00:00\"\n"
                "   }\n"
                "]"
                "โปรดตอบกลับมาเป็น JSON array ของ object ตามตัวอย่างข้างต้น"
            )

            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    [pil_image, prompt_text]
                )

                if response and hasattr(response, 'text') and response.text is not None:
                    full_response_text = response.text.strip()
                    match = re.search(r'```json\s*\[(.*)\]\s*```|\[(.*)\]', full_response_text, re.DOTALL)
                    json_string = None
                    if match:
                        if match.group(1):
                            json_string = f"[{match.group(1)}]"
                        elif match.group(2):
                            json_string = f"[{match.group(2)}]"
                    
                    if json_string:
                        try:
                            extracted_data = json.loads(json_string)
                            if not isinstance(extracted_data, list) or not all(isinstance(item, dict) for item in extracted_data):
                                raise ValueError("JSON response is not a list of dictionaries.")
                            
                            logging.info(f"         รูปภาพ {image_filename}: OCR สำเร็จและได้ข้อมูล JSON")
                            logging.debug(f"         รูปภาพ {image_filename}: ข้อมูลที่ดึงได้: {json_string[:200]}...")
                        except json.JSONDecodeError as e:
                            logging.error(f"         รูปภาพ {image_filename}: ไม่สามารถ parse JSON response ได้: {e}. Response: {full_response_text[:500]}")
                            extracted_data = None
                        except ValueError as e:
                            logging.error(f"         รูปภาพ {image_filename}: รูปแบบ JSON ไม่ถูกต้อง: {e}. Response: {full_response_text[:500]}")
                            extracted_data = None
                    else:
                        logging.warning(f"         รูปภาพ {image_filename}: ไม่พบโครงสร้าง JSON ที่ถูกต้องในข้อความตอบกลับ: {full_response_text[:500]}")
                        extracted_data = None

                else:
                    logging.warning(f"         รูปภาพ {image_filename}: การตอบกลับ Gemini ไม่มีเนื้อหา text หรือ text เป็น None")
                    extracted_data = None

                if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                    usage_metadata = {
                        'prompt_token_count': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_token_count': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                    logging.info(f"         รูปภาพ {image_filename} ใช้ Token: Prompt={usage_metadata.get('prompt_token_count', 0)}, Output={usage_metadata.get('candidates_token_count', 0)}, Total={usage_metadata.get('total_token_count', 0)}")
                else:
                    logging.warning(f"         รูปภาพ {image_filename}: ไม่พบข้อมูล usage_metadata ในการตอบกลับ API")
                    usage_metadata = None

                logging.info(f"ประมวลผลรูปภาพ {image_filename} เสร็จสิ้น")

            except Exception as gemini_e:
                logging.error(f"async_ocr_document_from_image ({image_filename}): ข้อผิดพลาดในการเรียก Gemini API: {gemini_e}")
                if hasattr(gemini_e, 'response') and gemini_e.response and hasattr(gemini_e.response, 'usage_metadata') and gemini_e.response.usage_metadata is not None:
                    try:
                        usage_metadata = {
                            'prompt_token_count': getattr(gemini_e.response.usage_metadata, 'prompt_token_count', 0),
                            'candidates_token_count': getattr(gemini_e.response.usage_metadata, 'candidates_token_count', 0),
                            'total_token_count': getattr(gemini_e.response.usage_metadata, 'total_token_count', 0)
                        }
                        logging.warning(f"         รูปภาพ {image_filename}: ใช้ Token ก่อนเกิดข้อผิดพลาด API: Prompt={usage_metadata.get('prompt_token_count', 0)}, Output={usage_metadata.get('candidates_token_count', 0)}, Total={usage_metadata.get('total_token_count', 0)}")
                    except Exception as inner_e:
                        logging.warning(f"         รูปภาพ {image_filename}: ไม่สามารถดึง usage_metadata จาก response error ได้: {inner_e}")
                        usage_metadata = None
                else:
                    logging.error(f"รายละเอียดข้อผิดพลาดการตอบกลับ Gemini API (ถ้ามี): {getattr(gemini_e, 'response', 'N/A')}")
                    usage_metadata = None
                extracted_data = None
                return image_path, extracted_data, usage_metadata

        except Exception as error:
            logging.critical(f"async_ocr_document_from_image ({image_filename}): ข้อผิดพลาดที่ไม่คาดคิดระหว่างการประมวลผล: {error}", exc_info=True)
            extracted_data = None
            usage_metadata = None
            return image_path, extracted_data, usage_metadata

        finally:
            if 'pil_image' in locals() and pil_image is not None:
                try:
                    pil_image.close()
                    del locals()['pil_image']
                except Exception as cleanup_e:
                    logging.debug(f"รูปภาพ {image_filename}: ข้อผิดพลาดในการทำความสะอาดอ็อบเจกต์ 'pil_image': {cleanup_e}")

        return image_path, extracted_data, usage_metadata

def get_sheets_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('sheets', 'v4', credentials=creds)
        logging.info("เชื่อมต่อ Google Sheets API สำเร็จแล้ว")
        return service
    except HttpError as err:
        logging.critical(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets API: {err}")
        print(f"\n!!!! เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets API: {err} โปรดตรวจสอบไฟล์ credentials.json และการตั้งค่า API !!!!")
        sys.exit(1)

def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('drive', 'v3', credentials=creds)
        logging.info("เชื่อมต่อ Google Drive API สำเร็จแล้ว")
        return service
    except HttpError as err:
        logging.critical(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Drive API: {err}")
        print(f"\n!!!! เกิดข้อผิดพลาดในการเชื่อมต่อ Google Drive API: {err} โปรดตรวจสอบไฟล์ credentials.json และการตั้งค่า API !!!!")
        sys.exit(1)

async def download_image_from_drive(
    drive_service: Any,
    file_id: str,
    file_name: str,
    destination_folder: str
) -> Optional[str]:
    """
    ดาวน์โหลดไฟล์จาก Google Drive ไปยังโฟลเดอร์ปลายทางที่ระบุ
    และคืนค่าพาธของไฟล์ที่ดาวน์โหลด
    """
    logging.info(f"กำลังดาวน์โหลดไฟล์ '{file_name}' (ID: {file_id}) จาก Google Drive...")
    file_path = os.path.join(destination_folder, file_name)

    try:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = await asyncio.to_thread(downloader.next_chunk)
            logging.debug(f"ดาวน์โหลด '{file_name}': {int(status.progress() * 100)}%")
        logging.info(f"ดาวน์โหลด '{file_name}' สำเร็จแล้ว ไปยัง: {file_path}")
        return file_path
    except HttpError as error:
        logging.error(f"ข้อผิดพลาดในการดาวน์โหลดไฟล์ '{file_name}' จาก Google Drive: {error}")
        return None
    except Exception as e:
        logging.error(f"ข้อผิดพลาดที่ไม่คาดคิดในการดาวน์โหลดไฟล์ '{file_name}': {e}", exc_info=True)
        return None

async def write_to_google_sheet(
    service: Any,
    spreadsheet_id: str,
    sheet_name: str,
    data: List[List[str]]
):
    if not data:
        logging.info("ไม่มีข้อมูลที่จะเขียนลง Google Sheet.")
        return

    try:
        result = await asyncio.to_thread(
            lambda: service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=f"{sheet_name}!A:A"
            ).execute()
        )
        values = result.get('values', [])
        next_row = len(values) + 1
        
        # ปรับ Header: ไม่มี 'รูป (Cropped)' แล้ว
        header = ["เบอร์อะไหล่", "ชื่ออะไหล่", "รุ่นรถ", "ปีรถ", "ราคาปลีก", "ราคาส่ง", "รูป (Original Drive URL)", "Time_stamp"]
        
        if len(values) == 0 or values[0] != header:
            await asyncio.to_thread(
                lambda: service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{sheet_name}!A1",
                    valueInputOption="RAW",
                    body={"values": [header]}
                ).execute()
            )
            if len(values) == 0:
                next_row += 1
            logging.info(f"เพิ่ม Header ใน Google Sheet '{sheet_name}' แล้ว")

        range_to_write = f"{sheet_name}!A{next_row}"
        
        body = {
            'values': data
        }
        
        result = await asyncio.to_thread(
            lambda: service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_to_write,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
        )
        logging.info(f"{result.get('updatedCells')} เซลล์ถูกเขียนลงใน Google Sheet '{sheet_name}' ตั้งแต่ '{range_to_write}'")
        print(f"✅ ข้อมูล OCR ถูกบันทึกลงใน Google Sheet '{sheet_name}' ตั้งแต่ '{range_to_write}' แล้ว")

    except HttpError as error:
        logging.error(f"ข้อผิดพลาดในการเขียนข้อมูลลง Google Sheet: {error}")
        print(f"❌ เกิดข้อผิดพลาดในการเขียนข้อมูลลง Google Sheet: {error}")
    except Exception as e:
        logging.error(f"ข้อผิดพลาดที่ไม่คาดคิดในการเขียนข้อมูลลง Google Sheet: {e}", exc_info=True)
        print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิดในการเขียนข้อมูลลง Google Sheet: {e}")

async def main():
    if not GEMINI_API_KEY:
        logging.critical("ไม่ได้ตั้งค่าคีย์ Gemini API ไม่สามารถดำเนินการได้ โปรดตั้งค่า GEMINI_API_KEY")
        print("\n!!!! ไม่ได้ตั้งค่าคีย์ Gemini API โปรดตั้งค่าตัวแปรสภาพแวดล้อม GEMINI_API_KEY ในไฟล์ key.env หรือ .env หรือในระบบของคุณ !!!!")
        sys.exit(1)
    
    if SPREADSHEET_ID == 'YOUR_SPREADSHEET_ID':
        logging.critical("ยังไม่ได้กำหนด SPREADSHEET_ID โปรดแก้ไขตัวแปร SPREADSHEET_ID ในโค้ด")
        print("\n!!!! ยังไม่ได้กำหนด SPREADSHEET_ID โปรดแก้ไขตัวแปร SPREADSHEET_ID ในโค้ด !!!!")
        sys.exit(1)

    GOOGLE_DRIVE_INPUT_FOLDER_ID = '1Vs9x28IA78S5ZL9g8S1f3xM9AskXLnVX' 

    gemini_model = None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("กำหนดค่า Gemini API สำเร็จแล้ว")
        model_name = 'gemini-2.5-flash'
        logging.info(f"กำลังโหลด Gemini GenerativeModel: {model_name}")
        gemini_model = genai.GenerativeModel(model_name=model_name)
        logging.info(f"โหลด Gemini GenerativeModel สำเร็จแล้ว: {gemini_model.model_name}")

        try:
            logging.info("กำลังตรวจสอบ Gemini API Connectivity...")
            await asyncio.to_thread(gemini_model.count_tokens, "test prompt")
            logging.info("ตรวจสอบ Gemini API Connectivity สำเร็จแล้ว")
        except Exception as api_check_e:
            logging.warning(f"ไม่สามารถตรวจสอบ Gemini API Connectivity: {api_check_e}. อาจมีปัญหากับการเรียก API จริง.")
            print(f"\nคำเตือน: ไม่สามารถตรวจสอบการเชื่อมต่อ Gemini API เบื้องต้นได้: {api_check_e}")

    except Exception as e:
        logging.critical(f"โหลด Gemini GenerativeModel ไม่สำเร็จ หรือตั้งค่า API มีปัญหา: {e} ออกจากโปรแกรม")
        print(f"\n!!!! โหลด Gemini GenerativeModel ไม่สำเร็จ หรือตั้งค่า API มีปัญหา: {e} โปรดตรวจสอบชื่อโมเดลและการตั้งค่า API Key !!!!")
        sys.exit(1)
    
    drive_service = get_drive_service()
    if not drive_service:
        sys.exit(1)

    logging.info(f"กำลังค้นหาไฟล์รูปภาพที่รองรับใน Google Drive Folder ID: {GOOGLE_DRIVE_INPUT_FOLDER_ID}")
    
    drive_image_files_info: List[Tuple[str, str, str]] = [] # (file_id, file_name, web_view_link)
    try:
        # ค้นหาไฟล์รูปภาพในโฟลเดอร์ที่ระบุ
        query = f"'{GOOGLE_DRIVE_INPUT_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"
        results = await asyncio.to_thread(
            lambda: drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, webViewLink)' # ดึง webViewLink มาด้วย
            ).execute()
        )
        items = results.get('files', [])
        for item in items:
            drive_image_files_info.append((item['id'], item['name'], item.get('webViewLink', 'N/A')))
        
    except HttpError as error:
        logging.critical(f"ข้อผิดพลาดในการค้นหาไฟล์ใน Google Drive: {error}")
        print(f"\n!!!! ข้อผิดพลาดในการค้นหาไฟล์ใน Google Drive: {error} โปรดตรวจสอบ ID โฟลเดอร์และสิทธิ์การเข้าถึง !!!!")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"ข้อผิดพลาดที่ไม่คาดคิดในการค้นหาไฟล์ใน Google Drive: {e}")
        print(f"\n!!!! ข้อผิดพลาดที่ไม่คาดคิดในการค้นหาไฟล์ใน Google Drive: {e} !!!!")
        sys.exit(1)

    if not drive_image_files_info:
        logging.warning(f"ไม่พบไฟล์รูปภาพที่รองรับใน Google Drive Folder ID: {GOOGLE_DRIVE_INPUT_FOLDER_ID} ไม่มีรูปภาพให้ประมวลผล")
        print(f"\nคำเตือน: ไม่พบไฟล์รูปภาพที่รองรับใน Google Drive Folder ID: {GOOGLE_DRIVE_INPUT_FOLDER_ID} ไม่มีรูปภาพให้ประมวลผล")
        return

    logging.info(f"พบไฟล์รูปภาพใน Google Drive {len(drive_image_files_info)} ไฟล์ พร้อมประมวลผล")

    print(f"\n=== เริ่มประมวลผล OCR เอกสารสำหรับ {len(drive_image_files_info)} รูปภาพ ===")
    logging.info(f"กำลังเริ่มประมวลผล OCR เอกสารสำหรับ {len(drive_image_files_info)} รูปภาพ (การทำงานพร้อมกันสูงสุด: {CONCURRENCY_OCR})")

    semaphore_ocr = asyncio.Semaphore(CONCURRENCY_OCR)
    ocr_tasks = []
    
    # ดาวน์โหลดภาพทั้งหมดก่อนเริ่มประมวลผล OCR
    downloaded_files_map = {} # {original_drive_url: local_path}
    for file_id, file_name, web_view_link in drive_image_files_info:
        local_path = await download_image_from_drive(drive_service, file_id, file_name, TEMP_DOWNLOAD_DIR)
        if local_path:
            downloaded_files_map[web_view_link] = local_path
            task = asyncio.create_task(
                async_ocr_document_from_image(
                    image_path=local_path,
                    model=gemini_model,
                    semaphore=semaphore_ocr,
                )
            )
            ocr_tasks.append((task, web_view_link)) # เก็บ URL ของ Drive ไว้ด้วย

    results_with_urls = await asyncio.gather(*[t[0] for t in ocr_tasks], return_exceptions=True)
    
    # สร้าง map จาก local_path ไปยัง original_drive_url
    local_path_to_drive_url = {v: k for k, v in downloaded_files_map.items()}

    logging.info("ประมวลผล OCR เอกสารแบบขนานเสร็จสิ้นแล้ว")
    print("=== ประมวลผลภาพเสร็จสิ้น กำลังรวบรวมผลลัพธ์ ===")

    logging.info("กำลังรวบรวมผลลัพธ์การ OCR เอกสารและคำนวณการใช้ Token...")

    document_ocr_results: List[Tuple[str, List[Dict[str, str]]]] = [] 
    sheets_data_to_write: List[List[str]] = []

    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_total_tokens = 0

    failed_images_count = 0
    successful_images_count = 0
    
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for i, result in enumerate(results_with_urls):
        original_drive_url = ocr_tasks[i][1] # ดึง URL ของ Drive จาก task ที่เราเก็บไว้
        local_image_path = ocr_tasks[i][0].result()[0] if not isinstance(result, Exception) else "N/A" # ดึง local path จาก result

        if isinstance(result, Exception):
            logging.error(f"Task ประมวลผลรูปภาพหนึ่งเกิดข้อผิดพลาด (Exception): {result}")
            failed_images_count += 1
            # เพิ่มข้อมูลลง sheet แม้จะล้มเหลว
            sheets_data_to_write.append([
                os.path.basename(local_image_path) if local_image_path != "N/A" else "N/A", # ใช้ชื่อไฟล์จาก local path
                "ไม่สามารถประมวลผลรูปภาพนี้ได้ หรือเกิดข้อผิดพลาด", 
                "", "", "", "", 
                original_drive_url, # URL ของ Drive
                current_time_str
            ])
            continue 

        image_path_original, extracted_data_content, usage_metadata_item = result
        image_filename = os.path.basename(image_path_original)
        
        if extracted_data_content is not None and extracted_data_content:
            document_ocr_results.append((image_filename, extracted_data_content))
            successful_images_count += 1

            for item in extracted_data_content:
                row_data = [
                    item.get("เบอร์อะไหล่", "N/A"),
                    item.get("ชื่ออะไหล่", "N/A"),
                    item.get("รุ่นรถ", "N/A"),
                    item.get("ปีรถ", "N/A"),
                    item.get("ราคาปลีก", "N/A"),
                    item.get("ราคาส่ง", "N/A"),
                    original_drive_url, # ใช้ URL ของ Drive ที่ดึงมา
                    current_time_str
                ]
                sheets_data_to_write.append(row_data)
        else:
            logging.warning(f"รูปภาพ {image_filename}: ไม่สามารถดึงข้อมูลอะไหล่ได้ หรือข้อมูลว่างเปล่า")
            sheets_data_to_write.append([
                image_filename, 
                "ไม่สามารถประมวลผลรูปภาพนี้ได้ หรือเกิดข้อผิดพลาด", 
                "", "", "", "", 
                original_drive_url, # URL ของ Drive
                current_time_str
            ])
            failed_images_count += 1

        if usage_metadata_item:
            total_prompt_tokens += usage_metadata_item.get('prompt_token_count', 0)
            total_candidates_tokens += usage_metadata_item.get('candidates_token_count', 0)
            total_total_tokens += usage_metadata_item.get('total_token_count', 0)

    logging.info("รวบรวมผลลัพธ์เสร็จสิ้น.")

    print(f"\n=== กำลังบันทึกผลลัพธ์ไปที่ไฟล์ '{OUTPUT_MARKDOWN_FILE}' ===")
    logging.info(f"กำลังบันทึกผลลัพธ์ทั้งหมดไปที่ไฟล์ '{OUTPUT_MARKDOWN_FILE}'...")
    try:
        if document_ocr_results:
            current_time = datetime.datetime.now()
            with open(OUTPUT_MARKDOWN_FILE, "w", encoding="utf-8") as f:
                f.write("# ผลลัพธ์ OCR เอกสารอะไหล่รถยนต์\n\n")
                f.write(f"ประมวลผลเมื่อ: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"โฟลเดอร์อินพุต Google Drive ID: `{GOOGLE_DRIVE_INPUT_FOLDER_ID}`\n\n")
                f.write("---\n\n")

                for filename, extracted_data_list in document_ocr_results:
                    f.write(f"## {filename}\n\n")
                    f.write("```json\n")
                    # ไม่ต้องลบ cropped_image_path เพราะไม่มีแล้ว
                    f.write(json.dumps(extracted_data_list, indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                    f.write("---\n\n")

            logging.info(f"บันทึกผลลัพธ์ทั้งหมดไปที่ '{OUTPUT_MARKDOWN_FILE}' แล้ว")
            print(f"\n✨ บันทึกผลลัพธ์ทั้งหมดไปที่ '{OUTPUT_MARKDOWN_FILE}' แล้ว ✨")
        else:
            logging.warning("ไม่มีผลลัพธ์ที่ประมวลผลได้สำเร็จ และไม่มีไฟล์รูปภาพให้ประมวลผล ไม่มีการสร้างไฟล์ Markdown")
            print("\nคำเตือน: ไม่มีผลลัพธ์ที่ประมวลผลได้สำเร็จ และไม่มีไฟล์รูปภาพให้ประมวลผล ไม่มีการสร้างไฟล์ Markdown")
    except Exception as e:
        logging.critical(f"ข้อผิดพลาดในการบันทึกไฟล์ Markdown '{OUTPUT_MARKDOWN_FILE}': {e}")
        print(f"\n!!!! ข้อผิดพลาดในการบันทึกไฟล์ Markdown '{OUTPUT_MARKDOWN_FILE}': {e} !!!!")

    if sheets_data_to_write:
        print("\n=== กำลังบันทึกผลลัพธ์ลง Google Sheet ===")
        logging.info("กำลังเชื่อมต่อและบันทึกผลลัพธ์ลง Google Sheet...")
        sheets_service = get_sheets_service()
        if sheets_service:
            await write_to_google_sheet(sheets_service, SPREADSHEET_ID, SHEET_NAME, sheets_data_to_write)
    else:
        logging.warning("ไม่มีข้อมูล OCR ที่จะบันทึกลง Google Sheet.")
        print("\nคำเตือน: ไม่มีข้อมูล OCR ที่จะบันทึกลง Google Sheet.")

    print("\n=== สรุปการประมวลผลและใช้ Token Gemini API ===")
    print(f"โฟลเดอร์อินพุต Google Drive ID: {GOOGLE_DRIVE_INPUT_FOLDER_ID}")
    print(f"ไฟล์รูปภาพที่พบใน Google Drive ทั้งหมด: {len(drive_image_files_info)}")
    print(f"จำนวน Task ที่พยายามรันประมวลผล: {len(ocr_tasks)}")
    print(f"ไฟล์รูปภาพที่ประมวลผลสำเร็จ (ดึงข้อความได้): {successful_images_count}")
    print(f"ไฟล์รูปภาพที่การประมวลผลกับ Gemini ล้มเหลว: {failed_images_count}")
    print("-" * 30)
    print(f"Token อินพุตทั้งหมด (Prompt + รูปภาพ): {total_prompt_tokens}")
    print(f"Token เอาต์พุตทั้งหมด (ข้อความที่สร้าง): {total_candidates_tokens}")
    print(f"Token ทั้งหมด: {total_total_tokens}")
    print("==========================================")
    logging.info(f"สรุปการประมวลผล: ไฟล์ทั้งหมด={len(drive_image_files_info)}, Task สร้าง={len(ocr_tasks)}, สำเร็จ/ไม่พบ={successful_images_count}, ล้มเหลว={failed_images_count}. การใช้ Token: Total={total_total_tokens}, Prompt={total_prompt_tokens}, Candidates={total_candidates_tokens}")

    if failed_images_count > 0:
        print(f"\nคำเตือน: มี {failed_images_count} ไฟล์ที่ประมวลผลไม่สำเร็จ ตรวจสอบ Log ด้านบนสำหรับรายละเอียด")

    # ลบไฟล์ที่ดาวน์โหลดชั่วคราวทั้งหมด
    if os.path.exists(TEMP_DOWNLOAD_DIR):
        for f in os.listdir(TEMP_DOWNLOAD_DIR):
            try:
                os.remove(os.path.join(TEMP_DOWNLOAD_DIR, f))
                logging.debug(f"ลบไฟล์ชั่วคราว: {f}")
            except OSError as e:
                logging.warning(f"ไม่สามารถลบไฟล์ชั่วคราว '{f}' ใน '{TEMP_DOWNLOAD_DIR}': {e}")
        try:
            os.rmdir(TEMP_DOWNLOAD_DIR)
            logging.info(f"ลบโฟลเดอร์ชั่วคราว '{TEMP_DOWNLOAD_DIR}' แล้ว")
        except OSError as e:
            logging.warning(f"ไม่สามารถลบโฟลเดอร์ชั่วคราว '{TEMP_DOWNLOAD_DIR}' (อาจจะไม่ว่างเปล่า): {e}")


if __name__ == "__main__":
    asyncio.run(main())