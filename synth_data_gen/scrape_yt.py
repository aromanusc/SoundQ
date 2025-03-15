import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import argparse
from datetime import datetime

def setup_youtube_api(api_key=None):
    """
    Set up the YouTube API client.
    If api_key is provided, use API key authentication.
    Otherwise, use OAuth 2.0 authentication.
    """
    if api_key:
        # Use API key authentication
        api_service_name = "youtube"
        api_version = "v3"
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=api_key)
    else:
        # Use OAuth 2.0 authentication
        # Get credentials and create an API client
        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        client_secrets_file = "client_secret.json"
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, scopes)
        credentials = flow.run_local_server(port=8080)
        api_service_name = "youtube"
        api_version = "v3"
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, credentials=credentials)
    
    return youtube

def search_videos(youtube, query, max_results=10):
    """
    Search YouTube for videos matching the given query.
    """
    # Call the search.list method to retrieve results matching the specified
    # query term.
    search_response = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=max_results,
        type="video",
        videoDuration="short"  # Focus on short videos which are more likely to contain specific sounds
    ).execute()

    videos = []
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videos.append({
                "title": search_result["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                "description": search_result["snippet"]["description"],
                "video_id": search_result["id"]["videoId"]
            })
    
    return videos

def get_video_duration(youtube, video_id):
    """
    Get the duration of a video in seconds.
    """
    video_response = youtube.videos().list(
        part="contentDetails",
        id=video_id
    ).execute()
    
    if not video_response.get("items"):
        return None
    
    duration_str = video_response["items"][0]["contentDetails"]["duration"]
    # Convert ISO 8601 duration to seconds (simplified)
    # This is a simplified implementation and may not handle all cases correctly
    duration_str = duration_str.replace("PT", "")
    seconds = 0
    if "H" in duration_str:
        hours, duration_str = duration_str.split("H")
        seconds += int(hours) * 3600
    if "M" in duration_str:
        minutes, duration_str = duration_str.split("M")
        seconds += int(minutes) * 60
    if "S" in duration_str:
        s = duration_str.replace("S", "")
        if s:
            seconds += int(s)
    return seconds

def main():
    parser = argparse.ArgumentParser(description='Search YouTube for sound event videos')
    parser.add_argument('--api_key', type=str, help='YouTube API key', required=False)
    parser.add_argument('--results', type=int, default=5, help='Number of results per class')
    parser.add_argument('--output', type=str, default='youtube_sound_events.csv', help='Output CSV file')
    args = parser.parse_args()
    
    # Set up YouTube API client
    youtube = setup_youtube_api(args.api_key)
    
    # Define sound event classes and search terms
    sound_classes = {
        0: ["female vlogger on camera", "woman speaking", "female talking", "woman talking on camera"],
        1: ["male vlogger on camera", "man speaking", "male talking", "man talking on camera"],
        2: ["clapping", "applause", "hand clapping", "person clapping"],
        3: ["telephone ringing", "phone ringing", "telephone sound", "phone call sound"],
        4: ["laughter", "people laughing", "baby laugh", "laughing person"],
        5: ["roomba", "vacuum cleaner", "kitchen dishes", "dish washing asmr"],
        6: ["person running", "walking person", "footsteps on floor", "walking on sand"],
        7: ["door opening", "door closing", "door asmr", "door creak"],
        8: ["loud speaker", "speaker", "loudspeaker bluetooth", "bluetooth speaker"],
        9: ["musical instrument", "instrument playing", "guitar", "piano", "violin", "drum"],
        10: ["water tap", "faucet asmr", "running water", "bathroom sink"],
        11: ["bike bell", "bell", "doorbell", "school bell"],
        12: ["knocking", "door knock", "knock table", "knocking door"]
    }
    
    # Create a DataFrame to store the results
    results = []
    
    # Search for videos for each class
    for class_id, search_terms in sound_classes.items():
        print(f"Searching for Class {class_id}: {search_terms[0]}")
        
        for term in search_terms:
            videos = search_videos(youtube, term, max_results=args.results)
            
            for video in videos:
                duration = get_video_duration(youtube, video["video_id"])
                if duration and duration <= 600:  # Limit to videos less than 10 minutes
                    start_time = "0:00"
                    # Estimate a reasonable end time based on video length
                    if duration <= 60:
                        end_time = f"0:{duration}"
                    else:
                        minutes = duration // 60
                        seconds = duration % 60
                        end_time = f"{minutes}:{seconds:02d}"
                    
                    results.append({
                        "link": video["link"],
                        "start": start_time,
                        "end": end_time,
                        "class": class_id,
                        "title": video["title"],
                        "description": video["description"]
                    })
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Keep only the required columns
    output_df = df[["link", "start", "end", "class"]]
    
    # Save to CSV
    output_df.to_csv(args.output, index=False)
    print(f"Saved {len(output_df)} results to {args.output}")
    
    # Also save a detailed version with titles and descriptions
    detailed_output = args.output.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_output, index=False)
    print(f"Saved detailed results to {detailed_output}")

if __name__ == "__main__":
    main()
