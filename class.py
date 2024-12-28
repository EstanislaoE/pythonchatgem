import streamlit as st 
import os 
import google.generativeai as genai 
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str
    content: str

class GeminiModel:
    """Handles all Gemini AI model operations"""
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.environ["Gemini_API_KEY"])
        
        self.generation_config = {
            "temperature": 1, 
            "top_p": 0.95, 
            "top_k": 64, 
            "max_output_tokens": 800, 
            "response_mime_type": "text/plain",
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            system_instruction="Act as an experienced blogger to promote things"
        )
    
    def get_response(self, prompt: str) -> Optional[str]:
        """Get response from Gemini model"""
        try:
            chat_session = self.model.start_chat()
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error getting response from Gemini: {str(e)}")
            return None

class ChatInterface:
    """Manages the chat interface and message history"""
    def __init__(self):
        self.gemini_model = GeminiModel()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize or get the session state for messages"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a new message to the chat history"""
        message = ChatMessage(role=role, content=content)
        st.session_state.messages.append({"role": message.role, "content": message.content})
    
    def display_messages(self):
        """Display all messages in the chat interface"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def process_user_input(self, prompt: str):
        """Process user input and get AI response"""
        if prompt:
            # Display and save user message
            st.chat_message("user").markdown(prompt)
            self.add_message("user", prompt)
            
            # Get and display AI response
            response = self.gemini_model.get_response(prompt)
            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)
                self.add_message("assistant", response)

def main():
    """Main application entry point"""
    st.header('HT header title')
    st.write("enter the topic on the write")
    
    # Initialize chat interface
    chat_interface = ChatInterface()
    
    # Display existing messages
    chat_interface.display_messages()
    
    # Handle user input
    if prompt := st.chat_input("Enter any topics"):
        chat_interface.process_user_input(prompt)

if __name__ == "__main__":
    main()