
from PIL import Image
import pytesseract

def find_best_ocr_rotation(image_path, angles=[0, 90, 180, 270], whitelist="0123456789"):
    """
    Rotates the image by each angle, runs OCR, and returns the best result (longest digit string).
    Returns: (best_text, best_angle)
    """
    img = Image.open(image_path)
    best_text = ""
    best_angle = 0
    for angle in angles:
        rotated = img.rotate(angle, expand=True)
        config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(rotated, config=config).strip()
        print(f"Angle {angle}: {text}")
        if len(text) > len(best_text):
            best_text = text
            best_angle = angle
    print("\nFinal OCR Result:", best_text)
    return best_text, best_angle

# Example usage:
if __name__ == "__main__":
    image_path = "detected_segments/new.jpg"
    find_best_ocr_rotation(image_path)
