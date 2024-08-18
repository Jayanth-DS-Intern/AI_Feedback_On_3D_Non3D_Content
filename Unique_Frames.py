import os
import cv2
import time
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import pickle

class UniqueFrames:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_or_save_model(self):
        model_path='vit_model.pkl'
        if os.path.exists(model_path):
            print("Loading saved model...")
            with open(model_path, 'rb') as f:
                self.processor, self.model = pickle.load(f)
        else:
            print("Initializing and saving new model...")
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            with open(model_path, 'wb') as f:
                pickle.dump((self.processor, self.model), f)

    def extract_frames(self, video_path, output_folder):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not video.isOpened():
            print("Error opening video file")
            return

        # Get the video frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"Video frame rate: {fps} FPS")

        # Initialize frame counter
        frame_count = 0

        # Read frames from the video
        while True:
            # Read a frame
            success, frame = video.read()

            # Break the loop if we have reached the end of the video
            if not success:
                break

            # Generate the output file name
            output_file = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

            # Save the frame as an image file
            cv2.imwrite(output_file, frame)

            # Increment the frame counter
            frame_count += 1

        # Release the video object
        video.release()

        print(f"Extracted {frame_count} frames to {output_folder}")

    # Example usage
    #extract_frames("/content/Inertia video.mp4", "/content/sample_data/Frames_Inertia")

    def process_frames_to_embeddings(self, frames_folder,batch_size=32):
        # # Check if CUDA is available and set the device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {device}")

        # # Load the ViT processor and model
        st = time.time()
        # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        # model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
        # model.eval()  # Set the model to evaluation mode

        if self.model is None or self.processor is None:
            self.load_or_save_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()

        # Get the expected image size from the processor
        expected_size = self.processor.size['height']  # Assuming square images

        # Get a list of all image files in the folder
        image_files = [f for f in sorted(os.listdir(frames_folder)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_frames = len(image_files)
        print("total_frames length--->",total_frames)
        print("image files length ---->",len(image_files))
        # Initialize a list to store all embeddings
        all_embeddings = []

        # Process the frames in batches
        for i in tqdm(range(0, total_frames, batch_size), desc="Processing batches"):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            for f in batch_files:
                try:
                    img = Image.open(os.path.join(frames_folder, f)).convert('RGB')
                    img = img.resize((expected_size, expected_size), Image.LANCZOS)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error processing image {f}: {e}")
                    continue

            if not batch_images:
                print("No valid images in this batch. Skipping.")
                continue

            try:
                # Process the batch of images and get embeddings
                batch_inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                
                with torch.no_grad():
                    batch_outputs = self.model(**batch_inputs)

                # Get the [CLS] token embeddings (first token of last hidden state)
                batch_embeddings = batch_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                print("batch embeddings------>",batch_embeddings)

                # Append the embeddings to the list
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        # Concatenate all embeddings
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            print("No embeddings were generated. Check your input data and processing steps.")
            return None

        et = time.time()
        print(f"Time taken to get the embeddings: {et-st:.2f} seconds")

        # if output_file:
        #     # Save embeddings to a file
        #     np.save(output_file, all_embeddings)
        #     print(f"Embeddings saved to {output_file}")

        print(f"Processed {len(all_embeddings)} frames.")
        print(f"Embeddings shape: {all_embeddings.shape}")

        return all_embeddings


    def select_unique_frames(self, embeddings, original_frames_folder, output_folder, similarity_threshold=0.90):
        """
        Select unique frames based on cosine similarity of their embeddings.

        :param embeddings: numpy array of frame embeddings
        :param original_frames_folder: folder containing original frame images
        :param output_folder: folder to store selected unique frames
        :param similarity_threshold: threshold below which frames are considered unique (default 0.2)
        :return: list of indices of selected unique frames
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        num_frames = embeddings.shape[0]
        unique_frame_indices = [0]  # Always include the first frame
        reference_embedding = embeddings[0]

        for i in range(1, num_frames):
            similarity = cosine_similarity(reference_embedding.reshape(1, -1), embeddings[i].reshape(1, -1))[0][0]

            if similarity < similarity_threshold:
                unique_frame_indices.append(i)
                reference_embedding = embeddings[i]

        # Copy unique frames to output folder
        frame_files = sorted([f for f in os.listdir(original_frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print("Frame_files------->",frame_files)
        print("-------------")
        for idx in unique_frame_indices:
            source_path = os.path.join(original_frames_folder, frame_files[idx])
            print("source_path-------->",source_path)
            print("-------------")
            dest_path = os.path.join(output_folder, frame_files[idx])
            print("dest_path-------->",dest_path)
            print("-------------")
            shutil.copy2(source_path, dest_path)
        print("unique_frames_indices-------->",unique_frame_indices)
        print(f"Selected {len(unique_frame_indices)} unique frames out of {num_frames} total frames.")
        return output_folder

    def main(self, video_3d_path):
        st = time.time()
        self.extract_frames(video_path=video_3d_path, output_folder='3d_video_frames')
        embeddings = self.process_frames_to_embeddings(frames_folder='3d_video_frames')
        if embeddings is not None:
            output_3d_unique_frames_folder = self.select_unique_frames(original_frames_folder='3d_video_frames', embeddings=embeddings, output_folder='Unique_3D_Frames')
            et = time.time()
            print("Time Taken to get embeddings----->", et-st)
            print("output_3d_unique_frames_folder----->", output_3d_unique_frames_folder)
            return output_3d_unique_frames_folder
        else:
            print("Failed to generate embeddings. Cannot proceed with unique frame selection.")
            return None

        

