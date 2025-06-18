import os
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
import base64
import io

from PIL import Image
import cv2
import numpy as np

# ตรวจสอบและนำเข้า python-dotenv อย่างปลอดภัย
try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False

import google.generativeai as genai
import json
import sys # สำหรับ sys.exit()

# --- การตั้งค่า Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- การโหลดตัวแปรสภาพแวดล้อม ---
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

# --- การตั้งค่าเพิ่มเติม: สำหรับการประมวลผลภาพและผลลัพธ์ ---
# >>>>> 1. ตั้งค่าพาธไปยังโฟลเดอร์รูปภาพเอกสารอินพุตของคุณ (กรุณาแก้ไข!) <<<<<
INPUT_IMAGE_DIRECTORY = r"C:\Users\topza\OneDrive\Desktop\work_pjk\OCR_part\img" # ตัวอย่าง

# >>>>> 2. ตั้งชื่อไฟล์ Markdown ผลลัพธ์ (กรุณาแก้ไขหากต้องการ) <<<<<
OUTPUT_MARKDOWN_FILE = "document_ocr_results.md" # เปลี่ยนชื่อไฟล์ผลลัพธ์เป็น .md

# >>>>> 3. จำนวน Task การประมวลผล Gemini ที่จะรันพร้อมกัน (ควบคุม Rate Limit API) <<<<<
CONCURRENCY_OCR = 5

# >>>>> 4. ตั้งค่า True หากต้องการบันทึกภาพที่ผ่านการปรับปรุงเบื้องต้นและปรับขนาดแล้ว <<<<<
SAVE_PREPROCESSED_IMAGES = True
# >>>>> 5. ไดเรกทอรีที่จะใช้บันทึกภาพที่ปรับปรุงแล้ว (กรุณาแก้ไขหากต้องการ) <<<<<
PREPROCESSED_OUTPUT_DIR = "preprocessed_document_output"
# ------------------------------------------------------------


# --- ฟังก์ชันการปรับขนาดรูปภาพเบื้องต้น ---
def resize_image(image: np.ndarray, target_width: int = 1500) -> Optional[np.ndarray]:
    """
    ปรับขนาดรูปภาพ (ในรูปแบบ OpenCV/NumPy array) หากความกว้างเกินกว่า target_width
    โดยรักษาสัดส่วนเดิม.

    Args:
        image: รูปภาพในรูปแบบ NumPy array (BGR format).
        target_width (int): ความกว้างสูงสุดที่ต้องการ.

    Returns:
        รูปภาพที่ปรับขนาดแล้ว หรือรูปภาพต้นฉบับหากไม่จำเป็นต้องปรับขนาด หรือ None หาก input เป็น None.
    """
    if image is None:
        logging.warning("resize_image: รูปภาพที่ป้อนเป็น None")
        return None
    h, w = image.shape[:2]
    if w > target_width:
        ratio = target_width / w
        new_height = int(h * ratio)
        if new_height <= 0 or target_width <= 0:
            logging.warning(f"resize_image: ขนาดใหม่ไม่ถูกต้อง ({target_width}, {new_height}) ข้ามการปรับขนาด")
            return image
        try:
            resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image
        except cv2.error as e:
            logging.error(f"resize_image: ข้อผิดพลาด CV2 ระหว่างการปรับขนาด: {e}")
            return image
    return image

# --- ฟังก์ชัน OCR เอกสารด้วย Gemini API สำหรับรูปภาพเดียว (Async Task) ---
async def async_ocr_document_from_image(
    image_path: str,
    model: genai.GenerativeModel,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str | None, Dict[str, int] | None]:
    """
    ประมวลผลไฟล์รูปภาพเดี่ยวแบบอะซิงโครนัสโดยใช้ Gemini API เพื่อ OCR เอกสารและดึงข้อความทั้งหมด.

    Args:
        image_path (str): เส้นทางไปยังไฟล์รูปภาพนำเข้า.
        model (genai.GenerativeModel): อินสแตนซ์โมเดล Gemini ที่กำหนดค่าแล้ว.
        semaphore (asyncio.Semaphore): Semaphore สำหรับควบคุมจำนวนการเรียก API พร้อมกัน เพื่อป้องกัน Rate Limit.

    Returns:
        Tuple[str, str | None, Dict[str, int] | None]:
            - ชื่อไฟล์รูปภาพที่ประมวลผล (ส่วนชื่อไฟล์, ไม่มีพาธ).
            - ข้อความทั้งหมดที่ OCR ได้ (string) หรือ None หากเกิดข้อผิดพลาดร้ายแรง.
            - ข้อมูลการใช้ Token (prompt_token_count, candidates_token_count, total_token_count) ในรูปแบบ Dictionary หรือ None.
    """
    image_filename = os.path.basename(image_path)
    if model is None:
        logging.error(f"async_ocr_document_from_image ({image_filename}): โมเดล Gemini เป็น None ข้ามการประมวลผล")
        return image_filename, None, None

    if not os.path.exists(image_path):
        logging.error(f"async_ocr_document_from_image ({image_filename}): ไม่พบไฟล์รูปภาพที่ {image_path}")
        return image_filename, None, None

    async with semaphore:
        logging.info(f"กำลังเริ่มประมวลผล OCR เอกสารสำหรับรูปภาพ: {image_filename}")
        extracted_text: str | None = None
        usage_metadata: Dict[str, int] | None = None

        try:
            cv_image = await asyncio.to_thread(cv2.imread, image_path)
            if cv_image is None:
                logging.error(f"async_ocr_document_from_image ({image_filename}): ไม่สามารถอ่านไฟล์รูปภาพได้: {image_path}")
                return image_filename, None, None

            current_image = cv_image

            # --- ขั้นตอนปรับปรุงภาพเบื้องต้นด้วย OpenCV (ยังคงมีประโยชน์สำหรับเอกสาร) ---
            try: # ปรับคอนทราสต์/ความสว่าง
                alpha = 1.3 # ลองค่า 1.0 - 2.0
                beta = 0   # ลองค่า -50 ถึง 50
                enhanced_image = cv2.convertScaleAbs(current_image, alpha=alpha, beta=beta)
                current_image = enhanced_image
                logging.debug(f"รูปภาพ {image_filename}: ทำการปรับปรุงคอนทราสต์/ความสว่างแล้ว")
            except Exception as e:
                logging.warning(f"รูปภาพ {image_filename}: ข้อผิดพลาดในการปรับปรุงคอนทราสต์/ความสว่าง: {e}")

            try: # เพิ่มความคมชัด
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened_image = cv2.filter2D(current_image, -1, kernel)
                current_image = sharpened_image
                logging.debug(f"รูปภาพ {image_filename}: ทำการเพิ่มความคมชัดแล้ว")
            except Exception as e:
                logging.warning(f"รูปภาพ {image_filename}: ข้อผิดพลาดในการเพิ่มความคมชัด: {e}")

            try: # ลดสัญญาณรบกวน (Denoising)
                denoised_image = cv2.fastNlMeansDenoisingColored(current_image, None, 10, 10, 7, 21)
                current_image = denoised_image
                logging.debug(f"รูปภาพ {image_filename}: ทำการลด Noise แล้ว")
            except Exception as e:
                logging.warning(f"รูปภาพ {image_filename}: ข้อผิดพลาดในการลด Noise: {e}")
            # --- สิ้นสุดขั้นตอนปรับปรุงภาพเบื้องต้น ---

            processed_image_cv = resize_image(current_image, target_width=1600) # อาจปรับ target_width ให้สูงขึ้นสำหรับเอกสาร
            if processed_image_cv is None:
                logging.error(f"async_ocr_document_from_image ({image_filename}): การปรับปรุงรูปภาพเบื้องต้นหรือปรับขนาดส่งผลให้เป็น None ข้ามการประมวลผล")
                return image_filename, None, None

            if SAVE_PREPROCESSED_IMAGES and processed_image_cv is not None:
                try:
                    await asyncio.to_thread(os.makedirs, PREPROCESSED_OUTPUT_DIR, exist_ok=True)
                    original_filename_base, original_filename_ext = os.path.splitext(image_filename)
                    if not original_filename_ext:
                        original_filename_ext = ".jpg"
                    save_filename = f"{original_filename_base}_preprocessed{original_filename_ext}"
                    save_path = os.path.join(PREPROCESSED_OUTPUT_DIR, save_filename)
                    await asyncio.to_thread(cv2.imwrite, save_path, processed_image_cv)
                    logging.info(f"รูปภาพ {image_filename}: บันทึกภาพที่ปรับปรุงแล้วที่ {save_path}")
                except Exception as save_e:
                    logging.warning(f"รูปภาพ {image_filename}: ไม่สามารถบันทึกภาพที่ปรับปรุงแล้วได้: {save_e}")

            try:
                pil_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB))
            except Exception as conv_e:
                logging.error(f"async_ocr_document_from_image ({image_filename}): ไม่สามารถแปลงรูปภาพ OpenCV เป็น PIL: {conv_e}")
                return image_filename, None, None

            logging.info(f"        กำลังเรียก Gemini API สำหรับรูปภาพ {image_filename} เพื่อ OCR เอกสาร...")

            # --- Prompt สำหรับ OCR เอกสารทั้งหมด ---
            # เน้นว่าเอกสารเป็นเรื่องอะไหล่รถยนต์ มีทั้งไทยและอังกฤษ
            # ขอผลลัพธ์เป็นข้อความธรรมดา แต่เน้นให้ระบุภาษาด้วยถ้าทำได้
            # โมเดล Gemini สามารถดึงข้อความทั้งหมดได้ดีโดยไม่ต้องให้ JSON format สำหรับการดึงข้อความดิบ
            # แต่ถ้าต้องการ metadata เพิ่มเติม เช่น ภาษา, ก็ยังคงขอ JSON ได้
            # สำหรับ Markdown เราจะดึงแค่ข้อความล้วนๆ มา แล้วมาจัดรูปแบบเอง
            prompt_text = (
                "จากภาพที่ให้มา เป็นเอกสารเกี่ยวกับอะไหล่รถยนต์ โปรดทำการ OCR "
                "และดึงข้อความทั้งหมดที่ปรากฏในเอกสาร ไม่ว่าจะเป็นภาษาไทยหรือภาษาอังกฤษ "
                "ให้ได้ครบถ้วนและถูกต้องที่สุดเท่าที่จะทำได้ "
                "ตามด้วยข้อความที่ดึงได้ทั้งหมด "
                "ห้ามมีข้อความหรือคำอธิบายอื่นใดเพิ่มเติมจากผลลัพธ์ OCR"
                #"ตัวอย่าง:\nLanguage: English\nPart Number: ABC-123\nDescription: Engine Filter\nPrice: $25.00"
                #"หรือ:\nLanguage: Thai\nรหัสอะไหล่: XYZ-456\nชื่อสินค้า: ไส้กรองน้ำมันเครื่อง\nราคา: 500 บาท"
            )


            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    [pil_image, prompt_text]
                )

                if response and hasattr(response, 'text') and response.text is not None:
                    extracted_text = response.text.strip()
                    logging.info(f"        รูปภาพ {image_filename}: OCR สำเร็จ")
                    logging.debug(f"        รูปภาพ {image_filename}: ข้อความที่ดึงได้: \"{extracted_text[:100]}...\"") # แสดง 100 ตัวแรก
                else:
                    logging.warning(f"        รูปภาพ {image_filename}: การตอบกลับ Gemini ไม่มีเนื้อหา text หรือ text เป็น None")
                    extracted_text = ""

                if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                    usage_metadata = {
                        'prompt_token_count': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_token_count': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                    logging.info(f"        รูปภาพ {image_filename} ใช้ Token: Prompt={usage_metadata.get('prompt_token_count', 0)}, Output={usage_metadata.get('candidates_token_count', 0)}, Total={usage_metadata.get('total_token_count', 0)}")
                else:
                    logging.warning(f"        รูปภาพ {image_filename}: ไม่พบข้อมูล usage_metadata ในการตอบกลับ API")
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
                        logging.warning(f"        รูปภาพ {image_filename} ใช้ Token ก่อนเกิดข้อผิดพลาด API: Prompt={usage_metadata.get('prompt_token_count', 0)}, Output={usage_metadata.get('candidates_token_count', 0)}, Total={usage_metadata.get('total_token_count', 0)}")
                    except Exception as inner_e:
                        logging.warning(f"        รูปภาพ {image_filename}: ไม่สามารถดึง usage_metadata จาก response error ได้: {inner_e}")
                        usage_metadata = None
                else:
                    logging.error(f"รายละเอียดข้อผิดพลาดการตอบกลับ Gemini API (ถ้ามี): {getattr(gemini_e, 'response', 'N/A')}")
                    usage_metadata = None
                extracted_text = None
                return image_filename, extracted_text, usage_metadata

        except Exception as error:
            logging.critical(f"async_ocr_document_from_image ({image_filename}): ข้อผิดพลาดที่ไม่คาดคิดระหว่างการประมวลผล: {error}", exc_info=True)
            extracted_text = None
            usage_metadata = None
            return image_filename, extracted_text, usage_metadata

        finally:
            for var_name in ['cv_image', 'current_image', 'processed_image_cv', 'pil_image', 'response']:
                if var_name in locals() and locals()[var_name] is not None:
                    try:
                        del locals()[var_name]
                    except Exception as cleanup_e:
                        logging.debug(f"รูปภาพ {image_filename}: ข้อผิดพลาดในการทำความสะอาดอ็อบเจกต์ '{var_name}': {cleanup_e}")

        return image_filename, extracted_text, usage_metadata


# --- ฟังก์ชันหลักสำหรับการ OCR เอกสาร (จัดการหลายไฟล์แบบ Async) ---
async def main():
    if not os.path.isdir(INPUT_IMAGE_DIRECTORY):
        logging.critical(f"ไม่พบโฟลเดอร์อินพุตที่: {INPUT_IMAGE_DIRECTORY} โปรดแก้ไขตัวแปร INPUT_IMAGE_DIRECTORY")
        print(f"\n!!!! ไม่พบโฟลเดอร์อินพุตที่: {INPUT_IMAGE_DIRECTORY} โปรดตรวจสอบและแก้ไขพาธในโค้ด !!!!")
        sys.exit(1)

    if not GEMINI_API_KEY:
        logging.critical("ไม่ได้ตั้งค่าคีย์ Gemini API ไม่สามารถดำเนินการได้ โปรดตั้งค่า GEMINI_API_KEY")
        print("\n!!!! ไม่ได้ตั้งค่าคีย์ Gemini API โปรดตั้งค่าตัวแปรสภาพแวดล้อม GEMINI_API_KEY ในไฟล์ key.env หรือ .env หรือในระบบของคุณ !!!!")
        sys.exit(1)

    gemini_model = None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("กำหนดค่า Gemini API สำเร็จแล้ว")
        model_name = 'gemini-1.5-pro-latest'
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

    logging.info(f"กำลังค้นหาไฟล์รูปภาพที่รองรับ (.png, .jpg, .jpeg) ในโฟลเดอร์: {INPUT_IMAGE_DIRECTORY}")
    image_files = []
    try:
        image_files = [
            os.path.join(INPUT_IMAGE_DIRECTORY, f)
            for f in os.listdir(INPUT_IMAGE_DIRECTORY)
            if os.path.isfile(os.path.join(INPUT_IMAGE_DIRECTORY, f))
               and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    except Exception as e:
        logging.critical(f"ข้อผิดพลาดในการอ่านรายชื่อไฟล์ในโฟลเดอร์ {INPUT_IMAGE_DIRECTORY}: {e}")
        print(f"\n!!!! ข้อผิดพลาดในการอ่านรายชื่อไฟล์ในโฟลเดORY: {e} !!!!")
        sys.exit(1)

    if not image_files:
        logging.warning(f"ไม่พบไฟล์รูปภาพที่รองรับ (.png, .jpg, .jpeg) ในโฟลเดอร์: {INPUT_IMAGE_DIRECTORY} ไม่มีรูปภาพให้ประมวลผล")
        print(f"\nคำเตือน: ไม่พบไฟล์รูปภาพที่รองรับในโฟลเดอร์ {INPUT_IMAGE_DIRECTORY} ไม่มีรูปภาพให้ประมวลผล")
        return

    logging.info(f"พบไฟล์รูปภาพที่เข้าเกณฑ์ {len(image_files)} ไฟล์ พร้อมประมวลผล")

    print(f"\n=== เริ่มประมวลผล OCR เอกสารสำหรับ {len(image_files)} รูปภาพ ===")
    logging.info(f"กำลังเริ่มประมวลผล OCR เอกสารสำหรับ {len(image_files)} รูปภาพ (การทำงานพร้อมกันสูงสุด: {CONCURRENCY_OCR})")

    semaphore_ocr = asyncio.Semaphore(CONCURRENCY_OCR)
    ocr_tasks = []
    for image_path in image_files:
        task = asyncio.create_task(
            async_ocr_document_from_image(
                image_path=image_path,
                model=gemini_model,
                semaphore=semaphore_ocr,
            )
        )
        ocr_tasks.append(task)

    results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
    logging.info("ประมวลผล OCR เอกสารแบบขนานเสร็จสิ้นแล้ว")
    print("=== ประมวลผลภาพเสร็จสิ้น กำลังรวบรวมผลลัพธ์ ===")

    # --- รวบรวมผลลัพธ์และคำนวณการใช้ Token ---
    logging.info("กำลังรวบรวมผลลัพธ์การ OCR เอกสารและคำนวณการใช้ Token...")

    # เราจะเก็บผลลัพธ์ในรูปแบบของ list ของ tuples เพื่อให้ง่ายต่อการสร้าง Markdown
    # แต่ละ tuple จะเป็น (image_filename, extracted_text_content)
    document_ocr_results: List[Tuple[str, str]] = []
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_total_tokens = 0

    failed_images_count = 0
    successful_images_count = 0

    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Task ประมวลผลรูปภาพหนึ่งเกิดข้อผิดพลาด (Exception): {result}")
            failed_images_count += 1
        else:
            image_filename, extracted_text_content, usage_metadata_item = result

            if extracted_text_content is not None:
                document_ocr_results.append((image_filename, extracted_text_content))
                successful_images_count += 1
            else:
                # ถ้าการประมวลผลในฟังก์ชัน async_ocr_document_from_image ล้มเหลว (คืนค่าเป็น None)
                document_ocr_results.append((image_filename, "ไม่สามารถประมวลผลรูปภาพนี้ได้ หรือเกิดข้อผิดพลาด"))
                failed_images_count += 1

            if usage_metadata_item:
                total_prompt_tokens += usage_metadata_item.get('prompt_token_count', 0)
                total_candidates_tokens += usage_metadata_item.get('candidates_token_count', 0)
                total_total_tokens += usage_metadata_item.get('total_token_count', 0)

    logging.info("รวบรวมผลลัพธ์เสร็จสิ้น.")

    # --- บันทึกผลลัพธ์ทั้งหมดไปที่ไฟล์ Markdown ---
    print(f"\n=== กำลังบันทึกผลลัพธ์ไปที่ไฟล์ '{OUTPUT_MARKDOWN_FILE}' ===")
    logging.info(f"กำลังบันทึกผลลัพธ์ทั้งหมดไปที่ไฟล์ '{OUTPUT_MARKDOWN_FILE}'...")
    try:
        if document_ocr_results:
            with open(OUTPUT_MARKDOWN_FILE, "w", encoding="utf-8") as f:
                f.write("# ผลลัพธ์ OCR เอกสารอะไหล่รถยนต์\n\n")
                f.write(f"ประมวลผลเมื่อ: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
                f.write(f"โฟลเดอร์อินพุต: `{INPUT_IMAGE_DIRECTORY}`\n\n")
                f.write("---\n\n")

                for filename, text_content in document_ocr_results:
                    f.write(f"## {filename}\n\n")
                    f.write("```\n") # ใช้ code block สำหรับข้อความที่ดึงมาเพื่อให้จัดรูปแบบง่าย
                    f.write(text_content.strip()) # strip() เพื่อลบช่องว่างหัวท้าย
                    f.write("\n```\n\n")
                    f.write("---\n\n") # คั่นระหว่างรูปภาพ

            logging.info(f"บันทึกผลลัพธ์ทั้งหมดไปที่ '{OUTPUT_MARKDOWN_FILE}' แล้ว")
            print(f"\n✨ บันทึกผลลัพธ์ทั้งหมดไปที่ '{OUTPUT_MARKDOWN_FILE}' แล้ว ✨")
        else:
            logging.warning("ไม่มีผลลัพธ์ที่ประมวลผลได้สำเร็จ และไม่มีไฟล์รูปภาพให้ประมวลผล ไม่มีการสร้างไฟล์ Markdown")
            print("\nคำเตือน: ไม่มีผลลัพธ์ที่ประมวลผลได้สำเร็จ และไม่มีไฟล์รูปภาพให้ประมวลผล ไม่มีการสร้างไฟล์ Markdown")
    except Exception as e:
        logging.critical(f"ข้อผิดพลาดในการบันทึกไฟล์ Markdown '{OUTPUT_MARKDOWN_FILE}': {e}")
        print(f"\n!!!! ข้อผิดพลาดในการบันทึกไฟล์ Markdown '{OUTPUT_MARKDOWN_FILE}': {e} !!!!")


    # --- แสดงสรุปการประมวลผลและใช้ Token ---
    print("\n=== สรุปการประมวลผลและใช้ Token Gemini API ===")
    print(f"โฟลเดอร์อินพุตที่ค้นหา: {INPUT_IMAGE_DIRECTORY}")
    print(f"ไฟล์รูปภาพที่พบทั้งหมด: {len(image_files)}")
    print(f"จำนวน Task ที่พยายามรันประมวลผล: {len(ocr_tasks)}")
    print(f"ไฟล์รูปภาพที่ประมวลผลสำเร็จ (ดึงข้อความได้): {successful_images_count}")
    print(f"ไฟล์รูปภาพที่การประมวลผลกับ Gemini ล้มเหลว: {failed_images_count}")
    print("-" * 30)
    print(f"Token อินพุตทั้งหมด (Prompt + รูปภาพ): {total_prompt_tokens}")
    print(f"Token เอาต์พุตทั้งหมด (ข้อความที่สร้าง): {total_candidates_tokens}")
    print(f"Token ทั้งหมด: {total_total_tokens}")
    print("==========================================")
    logging.info(f"สรุปการประมวลผล: ไฟล์ทั้งหมด={len(image_files)}, Task สร้าง={len(ocr_tasks)}, สำเร็จ/ไม่พบ={successful_images_count}, ล้มเหลว={failed_images_count}. การใช้ Token: Total={total_total_tokens}, Prompt={total_prompt_tokens}, Candidates={total_candidates_tokens}")

    if failed_images_count > 0:
        print(f"\nคำเตือน: มี {failed_images_count} ไฟล์ที่ประมวลผลไม่สำเร็จ ตรวจสอบ Log ด้านบนสำหรับรายละเอียด")

    if SAVE_PREPROCESSED_IMAGES:
        print(f"\nบันทึกภาพที่ผ่านการปรับปรุงแล้ว (ถ้ามี) ไว้ในโฟลเดอร์: ./{PREPROCESSED_OUTPUT_DIR}")


# --- จุดเริ่มต้นการทำงานของสคริปต์ ---
if __name__ == "__main__":
    import sys
    import datetime # นำเข้า datetime เพื่อใช้สำหรับบันทึกเวลา
    current_time = datetime.datetime.now() # ได้รับเวลาปัจจุบัน

    if sys.version_info < (3, 8):
        print("!!!! โปรดใช้ Python 3.8 ขึ้นไปเพื่อรองรับ asyncio.to_thread และไวยากรณ์ Union Type (str | None) !!!!")
        logging.critical("ต้องใช้ Python 3.8 ขึ้นไป")
        sys.exit(1)

    print("=== กำลังเริ่มต้นสคริปต์ OCR เอกสารด้วย Gemini API (Async) ===")
    logging.info("=== กำลังเริ่มต้นสคริปต์ OCR เอกสารด้วย Gemini API (Async) ===")
    try:
        asyncio.run(main())
        logging.info("สคริปต์ OCR เอกสารด้วย Gemini API เสร็จสมบูรณ์.")
        print("\n=== สคริปต์ประมวลผลเสร็จสมบูรณ์ ===")
    except FileNotFoundError as fnf_e:
        logging.critical(f"ข้อผิดพลาด: ไม่พบไฟล์หรือโฟลเดอร์ที่ระบุ: {fnf_e}")
        print(f"\n!!!! ข้อผิดพลาด: ไม่พบไฟล์หรือโฟลเดอร์ที่ระบุ: {fnf_e} โปรดตรวจสอบเส้นทาง !!!!")
        sys.exit(1)
    except ImportError as im_e:
        logging.critical(f"ข้อผิดพลาด: ไม่พบไลบรารีที่จำเป็น โปรดติดตั้ง: {im_e}")
        print(f"\n!!!! ข้อผิดพลาด: ไม่พบไลบรารีที่จำเป็น โปรดติดตั้ง: `pip install google-generativeai python-dotenv pillow opencv-python numpy` !!!!")
        sys.exit(1)
    except Exception as main_err:
        logging.critical(f"เกิดข้อผิดพลาดรุนแรงใน Main loop: {main_err}", exc_info=True)
        print(f"\n!!!! เกิดข้อผิดพลาดรุนแรงใน Main loop: {main_err} โปรดตรวจสอบ Log ด้านบนสำหรับรายละเอียด !!!!")
        sys.exit(1)