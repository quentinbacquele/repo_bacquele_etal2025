import pandas as pd
import requests
import csv
from collections import defaultdict
import random
import time
import os
from pathlib import Path
import subprocess
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class XenoCantoExtractor:
    def __init__(self, species_csv_path: str, output_csv: str, audio_folder: str):
        self.species_csv_path = species_csv_path
        self.output_csv = output_csv
        self.audio_folder = audio_folder
        self.base_url = 'https://xeno-canto.org/api/2/recordings'
        self.order_family_map = {}
        os.makedirs(audio_folder, exist_ok=True)

    def load_species(self) -> Set[str]:
        df = pd.read_csv(self.species_csv_path, usecols=['category', 'order', 'family', 'scientific name'])
        passerine_df = df[(df['category'] == 'species') & (df['order'] == 'Passeriformes')]
        self.order_family_map = passerine_df.set_index('scientific name')[['order', 'family']].to_dict('index')
        return set(passerine_df['scientific name'].drop_duplicates().tolist())

    def fetch_recordings(self) -> List[dict]:
        params = {'query': 'q:A type:song', 'page': 1}
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
        
        data = response.json()
        num_pages = int(data.get('numPages', 1))
        all_recordings = []
        
        for page in range(1, num_pages + 1):
            print(f"Fetching page {page}/{num_pages}")
            params['page'] = page
            response = requests.get(self.base_url, params=params)
            time.sleep(1)
            
            if response.status_code == 200:
                data = response.json()
                all_recordings.extend(data.get('recordings', []))
            
        return all_recordings

    def process_recordings(self, recordings: List[dict], species_set: Set[str]) -> Dict[str, List[dict]]:
        species_recordings = defaultdict(lambda: {'recordings': [], 'countries': set(), 'recordist_dates': set()})
        
        for rec in recordings:
            species_name = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
            if species_name not in species_set:
                continue
                
            duration = self.parse_duration(rec.get('length', ''))
            if not (3 <= duration < 60):
                continue
                
            recordist_date = f"{rec.get('rec', '')}_{rec.get('date', '')}"
            if recordist_date in species_recordings[species_name]['recordist_dates']:
                continue
                
            recording_data = self.prepare_recording_data(rec, species_name)
            species_recordings[species_name]['recordings'].append(recording_data)
            species_recordings[species_name]['countries'].add(rec.get('cnt', ''))
            species_recordings[species_name]['recordist_dates'].add(recordist_date)
        
        return species_recordings

    def balance_recordings(self, species_recordings: Dict[str, dict]) -> List[dict]:
        recording_counts = [len(data['recordings']) for data in species_recordings.values()]
        threshold = int(np.percentile(recording_counts, 90))
        print(f"Threshold: {threshold}")
        balanced_recordings = []

        species_counts = defaultdict(int)
        total_recordings = 0
        
        for species, data in species_recordings.items():
            count = len(data['recordings'])
            species_counts[count] += 1
            total_recordings += count
            
        print(f"Total recordings: {total_recordings}")
        print(f"Average per species: {total_recordings/len(species_recordings):.1f}")
        print(f"Species above threshold: {sum(1 for c in recording_counts if c > threshold)}")
            
        for species, data in species_recordings.items():
            if len(data['recordings']) < 3:
                continue
                
            recordings = data['recordings']
            if len(recordings) > threshold:
                countries = list(data['countries'])
                base_per_country = threshold // len(countries)
                
                country_recordings = defaultdict(list)
                for rec in recordings:
                    country_recordings[rec['country']].append(rec)
                
                selected = []
                countries_with_excess = []
                
                for country in countries:
                    country_recs = country_recordings[country]
                    random.shuffle(country_recs)
                    available = len(country_recs)
                    
                    if available >= base_per_country:
                        selected.extend(country_recs[:base_per_country])
                        if available > base_per_country:
                            countries_with_excess.append(country)
                    else:
                        selected.extend(country_recs)
                
                remaining = threshold - len(selected)
                if remaining > 0 and countries_with_excess:
                    extra_per_country = remaining // len(countries_with_excess)
                    final_extra = remaining % len(countries_with_excess)
                    
                    for country in countries_with_excess:
                        remaining_recs = [r for r in country_recordings[country] if r not in selected]
                        selected.extend(remaining_recs[:extra_per_country])
                    
                    if final_extra:
                        lucky_country = random.choice(countries_with_excess)
                        remaining_recs = [r for r in country_recordings[lucky_country] if r not in selected]
                        selected.extend(remaining_recs[:final_extra])
                
                balanced_recordings.extend(selected)
            else:
                balanced_recordings.extend(recordings)
        
        return balanced_recordings

    @staticmethod
    def parse_duration(length_str: str) -> int:
        try:
            parts = [int(part) for part in length_str.strip().split(':')]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except (ValueError, AttributeError):
            return 0
        return 0

    def prepare_recording_data(self, rec: dict, species_name: str) -> dict:
        file_url = rec.get('file', '')
        if file_url.startswith('//'):
            file_url = 'https:' + file_url
        elif not file_url.startswith('http'):
            file_url = 'https://www.xeno-canto.org' + file_url
        
        order_family = self.order_family_map.get(species_name, {'order': '', 'family': ''})

        return {
            'order': order_family['order'],
            'family': order_family['family'],
            'gen': rec.get('gen', ''),
            'species': species_name,
            'common_name': rec.get('en', ''),
            'ssp': rec.get('ssp', ''),
            'id': rec.get('id', ''),
            'recordist': rec.get('rec', ''),
            'date': rec.get('date', ''),
            'time': rec.get('time', ''),
            'country': rec.get('cnt', ''),
            'location': rec.get('loc', ''),
            'lat': rec.get('lat', None),
            'lng': rec.get('lng', None),
            'url': f"https://xeno-canto.org/{rec.get('id', '')}",
            'file': file_url,
            'type': rec.get('type', '').strip().lower(),
            'file_name': rec.get('file-name', ''),
            'length': rec.get('length', '')
        }

    @staticmethod
    def get_extension_from_content_type(content_type: str) -> str:
        return {
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav'
        }.get(content_type.lower())

    def process_audio(self, input_path: str, output_path: str, 
                     target_sample_rate: int = 44100, 
                     target_dBFS: float = -20.0) -> bool:
        try:
            command = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(target_sample_rate),
                '-ac', '1',
                '-filter:a', f'loudnorm=I={target_dBFS}',
                output_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Audio processing error: {e}")
            return False

    def download_audio(self, url: str, species_name: str, recording_id: str) -> Optional[str]:
        response = requests.get(url, stream=True, allow_redirects=True)
        if response.status_code != 200:
            return None

        content_type = response.headers.get('Content-Type', '').lower()
        ext = self.get_extension_from_content_type(content_type)
        if not ext:
            return None

        temp_path = os.path.join(self.audio_folder, f"temp_{species_name}_{recording_id}{ext}")
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            return None
            
        return temp_path

    @staticmethod
    def process_single_recording(args) -> Tuple[bool, str]:
        rec, audio_folder = args
        species_name = rec['species'].replace(' ', '_')
        target_path = os.path.join(audio_folder, f"{species_name}_{rec['id']}.wav")
        
        if os.path.exists(target_path):
            return True, f"Skipped existing: {species_name}_{rec['id']}"
            
        try:
            response = requests.get(rec['file'], stream=True, allow_redirects=True)
            if response.status_code != 200:
                return False, f"Download failed: {species_name}_{rec['id']}"

            content_type = response.headers.get('Content-Type', '').lower()
            ext = {
                'audio/mpeg': '.mp3',
                'audio/wav': '.wav',
                'audio/x-wav': '.wav'
            }.get(content_type.lower())
            
            if not ext:
                return False, f"Unsupported format: {species_name}_{rec['id']}"

            temp_path = os.path.join(audio_folder, f"temp_{species_name}_{rec['id']}{ext}")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(temp_path) == 0:
                os.remove(temp_path)
                return False, f"Empty file: {species_name}_{rec['id']}"

            command = [
                'ffmpeg',
                '-y',
                '-i', temp_path,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                '-filter:a', 'loudnorm=I=-20.0',
                target_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            os.remove(temp_path)
            return True, f"Processed: {species_name}_{rec['id']}"

        except Exception as e:
            return False, f"Error ({species_name}_{rec['id']}): {str(e)}"

    def download_and_process_audio(self, recordings: List[dict]) -> None:
        max_workers = min(32, os.cpu_count() * 2)
        args = [(rec, self.audio_folder) for rec in recordings]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_recording, arg) for arg in args]
            
            success_count = 0
            with tqdm(total=len(recordings), desc="Processing recordings") as pbar:
                for future in as_completed(futures):
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    pbar.set_postfix({'Success': f"{success_count}/{len(recordings)}"})
                    pbar.update(1)

        print(f"\nCompleted processing with {success_count}/{len(recordings)} successful conversions")

    def save_to_csv(self, recordings: List[dict]) -> None:
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            if not recordings:
                return
                
            writer = csv.DictWriter(f, fieldnames=recordings[0].keys())
            writer.writeheader()
            writer.writerows(recordings)

    def run(self):
        species_set = self.load_species()
        print(f"Loaded {len(species_set)} passerine species")
        
        recordings = self.fetch_recordings()
        print(f"Fetched {len(recordings)} total recordings")
        
        species_recordings = self.process_recordings(recordings, species_set)
        print(f"Found recordings for {len(species_recordings)} species out of {len(species_set)} passerine species ({(len(species_recordings)/len(species_set)*100):.1f}%)")
        
        species_counts = {species: len(data['recordings']) for species, data in species_recordings.items()}
        print("\nRecording distribution:")
        for count in sorted(set(species_counts.values()), reverse=True)[:10]:
            species_with_count = sum(1 for v in species_counts.values() if v == count)
            print(f"{count} recordings: {species_with_count} species")
        
        balanced_recordings = self.balance_recordings(species_recordings)
        print(f"\nSelected {len(balanced_recordings)} recordings after balancing")
        
        self.save_to_csv(balanced_recordings)
        self.download_and_process_audio(balanced_recordings)

def main():
    extractor = XenoCantoExtractor(
        species_csv_path='./data/Clements-v2024-October-2024-rev.csv',
        output_csv='./output/metadata_xeno_passerines.csv',
        audio_folder='./output/audio_files_passerines'
    )
    extractor.run()

if __name__ == "__main__":
    main()