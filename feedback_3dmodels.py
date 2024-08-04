import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
from anthropic import Anthropic
from system_tools import system_prompt,tools_3d_models_feedback
import tempfile
import streamlit as st
import json
 

feedback3d ,non3dfeedback = st.tabs(['3D Feedback', "Non 3D Feedback"])

def save_uploaded_file(uploaded_file):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            # Return the path of the temporary file
        return tmp_file.name

class AI_Feedback:
    def __init__(self):
        self.client_anthropic = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    #First, we use OpenCV to extract frames from the video
    def video_frames(self, video_path):
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate the number of frames to skip to get a frame every 0.5 seconds
        frames_to_skip = int(fps * 0.5)
        
        frames = []
        frame_count = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            # Extract a frame every 0.5 seconds
            if frame_count % frames_to_skip == 0:
                frames.append(frame)
            
            frame_count += 1
        
        video.release()
        
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Frames extracted: {len(frames)}")
        return frames

    def encode_base64(self,frames):
        base64Frames = []
        for frame in frames:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64Frames

    def send_request(self,system_prompt, message_list):
        try:
           # print("system_prompt------>",system_prompt)
           # print("message_List------->",message_list)
            response = self.client_anthropic.messages.create(
                                                model="claude-3-5-sonnet-20240620",
                                                max_tokens=4096,
                                                system=system_prompt,
                                                messages=message_list,
                                                tools=tools_3d_models_feedback,
                                                tool_choice={"type": "tool", "name": "3D_models_feedback"}
                                            )
            # response.raise_for_status()
            
            return response
        except Exception as e:
            print(f"Exception occurred in send request function as: {e}")

     

    def feedback_on_3d_Models(self, uploaded_video_path, topic):
        temp_video_path = save_uploaded_file(uploaded_video_path)
        s_t1 = time.time()
        Frames = self.video_frames(video_path=temp_video_path)
        e_d1 = time.time()
        print("Time taken for entire process------>",e_d1-s_t1)
        
        base64Frames = self.encode_base64(Frames)
        # Create the initial message with text content
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"These are the frames from a 3D model. <topic>{topic}</topic>"
                }
            ]
        }

        # Add image content to the same message
        for base64_string in base64Frames[0::15]:
            message["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_string
                }
            })

        # Create the message list with the single message
        message_list = [message]

        response = self.send_request(system_prompt=system_prompt, message_list=message_list)
        print("response---------->", response)
        print("___________------------___________")
        response_json = response.content[0].input
        function_response = json.dumps(response_json)
        function_response = json.loads(function_response)
        st.write(function_response['feedback'])
        return function_response

    def feedback_on_non_3dcontent():
        pass

     

 
def main():

    with(feedback3d):

        st.title("AI Feedback On 3D Models")

        # File uploader
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "webm"])
        
        if uploaded_video:
            temp_video_path = save_uploaded_file(uploaded_video)
            st.video(temp_video_path)
        else:
            st.write("pls upload the video")
            
        topic = st.text_area("Give the topic from which the 3d model created:",height=200,key='topic_3dcontent')
        if st.button("Get AI Feedback"): 
            feedback = AI_Feedback()
            feedback.feedback_on_3d_Models(uploaded_video,topic)

    with(non3dfeedback):
        st.write("soon")
        # st.title("AI Feedback On Non 3D Content")
        # upload_file = st.file_uploader("Upload a text file", type=['txt'])
        # topic_non3d = st.text_area("Give the topic from which the 3d model created:",height=200,key='topic_non3dcontent')

main()