# Steps to generate synthetic 360-degree audiovisual scapes

## 1 - Download the METU Sparg dataset

You can find the dataset at [https://zenodo.org/records/2635758](https://zenodo.org/records/2635758).

## 2 - Process the EM32 IRs

The RIRs are given as separate wavefiles for each of the 32 channels in the mic array. We need to join them into a single wavefile. In the codebase we use, each location has a wavefile, we call the joined wavfile as `IR_em32.wav`. Use the `remix_metu_rirs.py` script we provide.

## 3 - Download the video assets or collect them yourself from YouTube or other video libraries

### YouTube video scraping script: `scrape_yt.py`

This Python script will help you find YouTube videos that match your specified sound event classes. Here's how it works:

#### Features

- Searches YouTube for videos matching each of the 13 sound event classes you specified
- Uses the YouTube Data API to perform searches
- Provides timestamps for each video (start and end)
- Outputs results in the CSV format you requested
- Filters for shorter videos (under 10 minutes) that are more likely to contain clean sound examples

#### Setup Instructions

1. Install required packages:

```
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib pandas
```

2. You'll need YouTube API credentials. You can either:

 - Use an API key (simpler but rate-limited)
 - Set up OAuth 2.0 authentication (more complex but higher quotas)

3. For API key:

 - Go to the Google Cloud Console
 - Create a new project or select an existing one
 - Enable the YouTube Data API v3
 - Create an API key under "Credentials"

4. For OAuth (if you don't specify an API key):

 - Download the OAuth client configuration file as "`client_secret.json`"
 - Place it in the same directory as the script
 - Follow the authorization prompts when running the script


#### Usage

```
python scrape_yt.py --api_key YOUR_API_KEY --results 5 --output youtube_sound_events.csv
```

Parameters:

`--api_key`: Your YouTube API key (optional if using OAuth)
`--results`: Number of results to fetch per class (default: 5)
`--output`: Output CSV file name (default: `youtube_sound_events.csv`)

The script will create two files:

- A CSV file with just the link, start, end, and class (matching your format)
- A detailed CSV that includes video titles and descriptions

#### Data download

Run the download script pointing to your generated YT csv file

```
python download.py
```

## 4 - Execute audiovisual synthetic data generator

### Mic format
```
python audiovisual_synth.py mic
```

### EM32 format
```
python audiovisual_synth.py em32
```
