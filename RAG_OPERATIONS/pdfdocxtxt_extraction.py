import cv2
import pytesseract
import numpy as np
import re
import traceback
from pdf2image import convert_from_bytes
from docx import Document
import asyncio

class DocPdfImageExtractor:
    def __init__(self,tesseract_path:str=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',pop_path:str=r'C:/Users/Deepak J Bhat/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin'):
        self.tesseract_path=tesseract_path
        self.output_dir="frames"
        self.pop_path=pop_path


    async def pdf_extraction(self,file_buffer):
        try:
            pytesseract.tesseract_cmd=self.tesseract_path
            file_buffer.seek(0)
            images=await asyncio.to_thread(convert_from_bytes,file_buffer.read(),poppler_path=self.pop_path,fmt="jpeg")
            final_extracted_text=[]
            async def extract_text(img):
                return asyncio.to_thread(pytesseract.image_to_string,img)
            tasks=[extract_text(img) for img in images]
            final_extracted_text=asyncio.gather(*tasks)
            ocr_extracted_text="".join(final_extracted_text)
            ocr_extracted_clean_text=re.sub('[^\x20-\x7E\n]','',ocr_extracted_text)
            return ocr_extracted_clean_text
        except Exception as e:
            print("Exception occured in pdf text extraction",traceback.format_exc())
            raise e


    async def doc_extraction(self,file_buffer):
        try:
            file_buffer.seek(0)
            doc=Document(file_buffer)
            extracted_text=[para.text for para in doc.paragraphs if para.text.strip()]
            doc_text=" ".join(extracted_text)
            doc_clean_text=re.sub(r'[^\x20-\x7E\n]','',doc_text)
            return doc_clean_text
        except Exception as e:
            print("Exception occured in doc text extraction",traceback.format_exc())
            raise e

    async def image_text_extraction(self,file_buffer):
        try:
            file_buffer.seek(0)
            file_bytes = np.frombuffer(file_buffer.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from buffer.")
            pytesseract.tesseract_cmd=self.tesseract_path
            gray_image=await asyncio.to_thread(cv2.cvtColor,image,cv2.COLOR_BGR2GRAY)
            grayscale_enhanced=await asyncio.to_thread(cv2.equalizeHist,gray_image)
            blurred=await asyncio.to_thread(cv2.GaussianBlur,grayscale_enhanced,(5,5),0)
            _,binary = await asyncio.to_thread(cv2.threshold,blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if binary is not None:
                text=await asyncio.to_thread(pytesseract.image_to_string,binary)
                cleaned_text=re.sub('[^\x20-\x7E\n]','',text)
                return cleaned_text
            else:
                return "No text extracted"
        except Exception as e:
            print("Exception occured in image text extraction",traceback.format_exc())
            raise e