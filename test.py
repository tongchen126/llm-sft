import base64
import time
from openai import OpenAI
from typing import Optional, List, Dict, Union, Any


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    return image_base64


def gpt_api(
    model: str,
    user: Optional[str] = None,
    system: Optional[str] = None,
    image_path: Optional[str] = None,
    history: Optional[List[Dict]] = None,
    retry_times: int = 5
) -> Dict[str, Any]:
    """
    Call GPT API with support for fresh or continued conversations.
    
    Args:
        model: Model name (e.g., "gpt-4", "gpt-4-vision")
        user: User message text
        system: System message (only used for fresh conversations)
        image_path: Path to image file (optional)
        history: Previous conversation messages (optional)
        retry_times: Number of retry attempts
        
    Returns:
        dict: {
            'text': str,           # Text response
            'images': list,        # List of image URLs (if any)
            'history': list        # Updated conversation history
        }
    """
    token = "irk4CnzkwB6dCF8VOOBxI2V3@2700"
    url = "http://v2.open.venus.oa.com/llmproxy"

    # Build messages
    if history is not None:
        # Continue existing conversation
        messages = history.copy()
    else:
        # Start fresh conversation
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
    
    # Add new user message if provided
    if user or image_path:
        user_content = []
        
        # Add text
        if user:
            user_content.append({"type": "text", "text": user})
        
        # Add image
        if image_path:
            image_base64 = image_to_base64(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })
        
        # Format message based on content
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            # Simple text-only message
            messages.append({"role": "user", "content": user})
        else:
            # Multimodal message
            messages.append({"role": "user", "content": user_content})

    # Make API call with retry logic
    client = OpenAI(base_url=url, api_key=token)
    
    attempts = retry_times
    while True:
        try:
            attempts -= 1
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            break
        except Exception as e:
            if attempts <= 0:
                raise Exception(f"Max retry reached. Last error: {str(e)}")
            time.sleep(2)

    # Parse response
    message = response.choices[0].message
    result = {
        'text': '',
        'images': [],
        'history': messages.copy()
    }
    
    # Extract content
    if isinstance(message.content, str):
        result['text'] = message.content
        # Add assistant response to history
        result['history'].append({"role": "assistant", "content": message.content})
        
    elif isinstance(message.content, list):
        # Handle multimodal response
        text_parts = []
        for item in message.content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'image_url' or 'multimodal' in item.get('type'):
                    result['images'].append(item.get(item.get('type'), {}).get('url', ''))
        
        result['text'] = ''.join(text_parts)
        # Add assistant response to history
        result['history'].append({"role": "assistant", "content": message.content})
    
    return result


def save_base64_image(base64_string: str, output_path: str):
    """Helper function to save base64 image to file"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(image_data)

def generate_multiple_images(keyword, model, save_dir, prompt='Please generate an image based on the keyword', repeat_num=10, save_start_index = 0):
    """
    Generate multiple images based on a keyword and save them to disk.
    
    Args:
        keyword: The keyword to base image generation on
        model: Model name (e.g., "dall-e-3", "dall-e-2")
        save_dir: Base directory to save images
        prompt: Prompt template for image generation
        repeat_num: Number of images to generate
        save_start_index: the index that saved images begin, suitable for continuing previous generation. 
    """
    import os
    import requests
    
    # Create directory structure: save_dir/keyword/
    output_dir = os.path.join(save_dir, keyword)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate images
    success_count = 0
    for i in range(repeat_num):
        try:
            # Combine prompt with keyword
            full_prompt = f"{prompt}: {keyword}"
            
            print(f"Generating image {i+1}/{repeat_num}...")
            
            # Call GPT API to generate image
            result = gpt_api(
                model=model,
                user=full_prompt
            )
            
            # Check if images were generated
            if result['images'] and len(result['images']) > 0:
                image_data = result['images'][0]
                
                # Determine output path
                output_path = os.path.join(output_dir, f"{keyword}_{i + save_start_index}.png")
                
                # Check if it's a base64 string or URL
                if image_data.startswith('http://') or image_data.startswith('https://'):
                    # Download from URL
                    response = requests.get(image_data)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Save base64 image
                    save_base64_image(image_data, output_path)
                
                print(f"✓ Successfully saved image {i+1}/{repeat_num} to {output_path}")
                success_count += 1
            else:
                print(f"✗ Warning: No image generated for iteration {i+1}")
                
        except Exception as e:
            print(f"✗ Error generating image {i+1}/{repeat_num}: {str(e)}")
            continue
        
        # Add small delay to avoid rate limiting
        if i < repeat_num - 1:
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{repeat_num} images successfully generated")
    print(f"Location: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir

# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    # Demo: generate multiple image given keyword and save to 'tmp/'
    for keyword in ['喜欢','开心','快乐','期待','高兴','痛苦']:
        generate_multiple_images(keyword, "gemini-2.5-flash-image", 'output_dir', \
                        prompt='Please generate an image based on the following keyword', repeat_num=3)

    # Demo: simple chat.
    response = gpt_api(
        model="gpt-5-chat",
        system="You are an image analysis assistant.",
        user="What's the result of 8 * 8 + 5?",
    )
    print(response['text'])

    # Demo: chat with image input.
    response = gpt_api(
        model="gpt-5-chat",
        system="You are an image analysis assistant.",
        user="Describe this image in detail.",
        image_path="input.jpg"  # Replace with actual path
    )
    print(response['text'])

    # Demo: chat with history conversation.
    response = gpt_api(
        model="gpt-5-chat",
        user="What else could you observe from it?",
        history=response['history']  # Pass the history from previous turn
    )
    print(response['text'])

    # Demo: ask GPT to generate image and save it.
    response = gpt_api(
        model="gemini-2.5-flash-image",  # Or nano-banana, gpt5, etc.
        user="Generate an image of a cute robot cat"
    )
    print(f"{response['text']}")
    print(f"{len(response['images'])}")
    
    if response['images']:
        for i, img_url in enumerate(response['images']):
            print(f"Image {i+1}: {img_url[:50]}...")
            # Save the image
            save_base64_image(img_url, f"output_{i}.png")
