import os
import torch
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import logging
from datetime import datetime
import traceback
from aegan_model import _RefinerG  # Import your model architecture
from esrgan_model import RRDBNet
import torchvision.transforms as transformss
from PIL import ImageFilter, ImageOps  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AEGANInference:
    def __init__(self, model_dir='pickle', upscale_factor=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aegan_model = self._load_aegan(os.path.join(model_dir, 'refinerG.pth'))
        self.esrgan_model = self._load_esrgan(os.path.join(model_dir, 'RRDB_ESRGAN_x4.pth'))
        self.upscale_factor = upscale_factor
        logger.info(f"Models loaded on {self.device}")

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_aegan(self, path):
        model = _RefinerG(nc=3, ngf=32)
        state_dict = torch.load(path, map_location=self.device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def _load_esrgan(self, path):
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device).eval()
        return model

    def preprocess_image(self, image_bytes):
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img = img.resize((512, 512), Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)
            img_tensor = torch.FloatTensor(img_array) / 255.0
            img_tensor = img_tensor * 2 - 1
            return img_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def postprocess_image(self, tensor):
        try:
            output = tensor.squeeze(0).cpu().detach()
            output = (output + 1) / 2
            output_np = output.clamp(0, 1).numpy().transpose(1, 2, 0)
            return Image.fromarray((output_np * 255).astype(np.uint8))
            #return img
        except Exception as e:
            logger.error(f"Image postprocessing failed: {str(e)}")
            raise

    # def enhance_image(self, image_bytes, mode='aegan'):
    #     """
    #     Enhancement Pipeline
        
    #     Args:
    #         image_bytes (bytes): Raw image bytes to enhance
    #         mode (str): 'aegan' for base model, 'esrgan' for super-res enhancement
    #     Returns:
    #         tuple: (output_path, message)
    #     """
    #     try:
    #         logger.info(f"Starting image enhancement in mode: {mode}")
    #         img_tensor = self.preprocess_image(image_bytes)
    #         # Step 1: AEGAN enhancement
    #         with torch.no_grad():
    #             refined_tensor = self.aegan_model(img_tensor)
    #         result_img = self.postprocess_image(refined_tensor)


    #         if mode == 'esrgan':
    #             # Step 2: ESRGAN super-resolution
    #             esrgan_input = transforms.ToTensor()(result_img).unsqueeze(0).to(self.device)
    #             with torch.no_grad():
    #                 output_tensor = self.esrgan_model(esrgan_input)
    #             output_np = output_tensor.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    #             result_img = Image.fromarray((output_np * 255).astype(np.uint8))

    #             # Post-ESRGAN refinement
    #             result_img = ImageOps.autocontrast(result_img)                          # auto contrast stretch
    #             result_img = ImageEnhance.Contrast(result_img).enhance(1.3)            # stronger contrast
    #             result_img = ImageEnhance.Sharpness(result_img).enhance(2.0)           # high sharpening
    #             result_img = result_img.filter(ImageFilter.EDGE_ENHANCE_MORE)          # edge enhancement


    #         # Save final image
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"enhanced_{mode}_{timestamp}.png"
    #         result_img.save(filename)
    #         logger.info(f"Saved enhanced image: {filename}")
    #         return filename, "Success"

        
            # else:
            #     return None, f"Unknown enhancement mode: {mode}"

            # # Save output
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # output_path = f"enhanced_{mode}_{timestamp}.png"
            # enhanced_img.save(output_path)

            # logger.info(f"Enhancement ({mode}) saved to {output_path}")
            # return output_path, f"Enhancement successful ({mode})"

        # except Exception as e:
        #     error_msg = f"Enhancement failed: {str(e)}"
        #     logger.error(error_msg)
        #     logger.error(traceback.format_exc())
        #     return None, error_msg  

    # def enhance_image(self, image_bytes):
    #     """
    #     Runs full enhancement pipeline: AEGAN → ESRGAN ×2 → filters
    #     Returns:
    #         tuple: (output_path, message)
    #     """
    #     try:
    #         logger.info("Starting full image enhancement pipeline (AEGAN + ESRGANx2 + filters)")

    #     # Preprocess input image
    #         img_tensor = self.preprocess_image(image_bytes)

    #     # Step 1: AEGAN enhancement
    #         with torch.no_grad():
    #             refined_tensor = self.aegan_model(img_tensor)

    #     # Convert AEGAN output to image
    #         aegan_output_img = self.postprocess_image(refined_tensor)

    #     # Step 2: ESRGAN super-resolution ×2
    #         esrgan_input = transforms.ToTensor()(aegan_output_img).unsqueeze(0).to(self.device)

    #         with torch.no_grad():
    #             first_pass = self.esrgan_model(esrgan_input)
    #             second_pass = self.esrgan_model(first_pass)
    #             output_tensor = second_pass

    #     # Convert to image
    #         output_np = output_tensor.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    #         result_img = Image.fromarray((output_np * 255).astype(np.uint8))

    #     # Step 3: Final enhancement filters
    #         from PIL import ImageFilter, ImageOps
    #         result_img = ImageOps.autocontrast(result_img)
    #         result_img = ImageEnhance.Contrast(result_img).enhance(1.3)
    #         result_img = ImageEnhance.Sharpness(result_img).enhance(2.0)
    #         result_img = result_img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    #     # Save final image
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"enhanced_final_{timestamp}.png"
    #         result_img.save(filename)

    #         logger.info(f"Saved final enhanced image to: {filename}")
    #         return filename, "Enhancement successful (AEGAN + ESRGAN + filters)"

    #     except Exception as e:
    #         error_msg = f"Enhancement failed: {str(e)}"
    #         logger.error(error_msg)
    #         logger.error(traceback.format_exc())
    #         return None, error_msg

    def enhance_image(self, image_bytes):
        """
        Robust enhancement: AEGAN → ESRGAN → Color-safe postprocessing
        """
        try:
            logger.info("Starting image enhancement pipeline (color-safe)")

        # Step 1: Load and preprocess image
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img = img.resize((512, 512), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]
            img_np = img_np.transpose(2, 0, 1)  # HWC → CHW
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

        # Normalize to [-1, 1] for AEGAN
            aegan_input = img_tensor * 2 - 1

        # Step 2: AEGAN inference
            with torch.no_grad():
                aegan_output = self.aegan_model(aegan_input)

        # Convert back from [-1, 1] to [0, 1] for ESRGAN
            esrgan_input = (aegan_output + 1) / 2.0

        # Step 3: ESRGAN inference (single pass to avoid artifacts)
            with torch.no_grad():
                sr_tensor = self.esrgan_model(esrgan_input)

            sr_tensor = sr_tensor.squeeze().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            final_img = Image.fromarray((sr_tensor * 255).astype(np.uint8))

        # Step 4: Color-safe postprocessing (no gamma or saturation shift)
            final_img = ImageEnhance.Sharpness(final_img).enhance(1.3)
            final_img = ImageEnhance.Contrast(final_img).enhance(1.1)

        # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"enhanced_final_{timestamp}.png"
            final_img.save(output_path)

            logger.info(f"Saved enhanced image to: {output_path}")
            return output_path, "Enhancement successful"

        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None, f"Enhancement failed: {str(e)}"


    @staticmethod
    def cleanup_temp_files(file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {str(e)}")


# Example usage:
if __name__ == "__main__":
    enhancer = AEGANInference(model_dir="models/latest")

    with open("test_image.jpg", "rb") as f:
        image_bytes = f.read()

    for mode in ['aegan', 'esrgan']:
        output_path, msg = enhancer.enhance_image(image_bytes, mode=mode)
        if output_path:
            print(f"[{mode.upper()}] Success! Saved to: {output_path}")
        else:
            print(f"[{mode.upper()}] Error: {msg}")
