import cv2
import numpy as np

def draw_annotations(image: np.ndarray, bbox: list[int] | None, label: str) -> np.ndarray:
    """
    Draw a bounding box and label on the image.
    bbox: [x1, y1, x2, y2] normalized to [1000, 1000] scale (Qwen2.5-VL format)
    """
    out = image.copy()
    h, w = out.shape[:2]
    
    # Neon Red #FF0000 (BGR = 0, 0, 255)
    neon_red = (0, 0, 255)
    thickness = 4
    
    if bbox and len(bbox) == 4:
        # Scale back from [1000, 1000] to original [w, h]
        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)
        
        # Clamp to bounds
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        
        cv2.rectangle(out, (x1, y1), (x2, y2), neon_red, thickness)
        
        # Task B5: Get text size
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        
        pad_x, pad_y = 6, 6
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2
        
        # Task B4: Intelligent positioning
        # Default: above the box, aligned to left edge
        text_x1 = x1
        text_y1 = y1 - box_h - 5
        
        # Top Edge Check: if too close to top, move inside/below the box
        if text_y1 < 5:
            text_y1 = y1 + 5
            # If the box itself is so small it goes out the bottom, adjust
            if text_y1 + box_h > h - 5:
                text_y1 = h - box_h - 5
                
        # Right Edge Check: if text extends past right, right-align it
        if text_x1 + box_w > w - 5:
            text_x1 = max(5, x2 - box_w)
            # If still extending (bbox is very wide but starts late), hard cap
            if text_x1 + box_w > w - 5:
                text_x1 = w - box_w - 5
                
        # Left edge safe zone
        text_x1 = max(5, text_x1)
        
        # Task B5: Solid filled rectangle for contrast
        cv2.rectangle(out, (text_x1, text_y1), (text_x1 + box_w, text_y1 + box_h), neon_red, -1)
        cv2.putText(out, label, (text_x1 + pad_x, text_y1 + th + pad_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        # Fallback to "Full Frame Scan"
        fallback_label = "Full Frame Scan"
        (tw, th), _ = cv2.getTextSize(fallback_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(out, fallback_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon_red, 2)
        
    return out
