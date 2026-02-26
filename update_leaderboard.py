import os
import subprocess
import pandas as pd
from io import StringIO
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import unpad

def get_git_file_info():
    """Identifies files added in the last merge and extracts their commit time."""
    try:
        raw_diff = subprocess.check_output(
            ['git', 'diff', '--name-status', 'HEAD^', 'HEAD'], 
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        
        changes = []
        for line in raw_diff.splitlines():
            if not line.strip(): continue
            status, path = line.split(None, 1)
            
            if path.endswith('.enc') and path.startswith('submissions/'):
                # NEW: Extract the Unix Author Timestamp (%at) for the specific file
                ts_cmd = ['git', 'log', '-1', '--format=%at', '--', path]
                timestamp_raw = subprocess.check_output(ts_cmd).decode('utf-8').strip()
                
                # NEW: Convert to readable UTC string
                sub_time = pd.to_datetime(int(timestamp_raw), unit='s').strftime('%Y-%m-%d %H:%M UTC')
                
                changes.append({
                    'path': path, 
                    'status': status, 
                    'sub_time': sub_time # Store the commit time
                })
        return changes
    except Exception as e:
        print(f"Git error: {e}")
        return []

def decrypt_file(encrypted_blob, private_key_str):
    """Decrypts a .enc file using RSA and AES-CBC (matching your evaluate.py logic)."""
    try:
        private_key = RSA.import_key(private_key_str.strip())
        
        # 1. Extract session key size (first 2 bytes)
        enc_session_key_size = int.from_bytes(encrypted_blob[:2], byteorder='big')
        # 2. Extract encrypted session key
        enc_session_key = encrypted_blob[2:2+enc_session_key_size]
        # 3. Extract IV (16 bytes)
        iv = encrypted_blob[2+enc_session_key_size : 2+enc_session_key_size+16]
        # 4. Extract Ciphertext
        ciphertext = encrypted_blob[2+enc_session_key_size+16:]
        
        # RSA Decrypt
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        
        # AES Decrypt
        cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
        decrypted_raw = unpad(cipher_aes.decrypt(ciphertext), AES.block_size)
        
        return decrypted_raw.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

def calculate_mae(gt_df, pred_df):
    """Calculates MAE with 8 decimal precision."""
    # Standardize column names
    gt_df.columns = gt_df.columns.str.lower().str.strip()
    pred_df.columns = pred_df.columns.str.lower().str.strip()
    
    merged = pd.merge(gt_df, pred_df, on='subject_session', suffixes=('_gt', '_pred'))
    if merged.empty:
        return None
    
    mae = (merged['age_at_visit_gt'] - merged['age_at_visit_pred']).abs().mean()
    return float(mae) # Keep as float, formatting happens at output

# --- 1. CONFIGURATION ---
gt_data = os.getenv('TEST_LABELS')
priv_key = os.getenv('RSA_PRIVATE_KEY')

if not gt_data or not priv_key:
    print("❌ Error: Missing Secrets.")
    exit(1)

gt_df = pd.read_csv(StringIO(gt_data))

# --- 2. SECURITY & FILE SELECTION ---
changes = get_git_file_info()

# Check for tampering (Modified or Deleted files in submissions/)
tampered = [c['path'] for c in changes if c['status'] != 'A']
if tampered:
    print(f"❌ SECURITY ERROR: The following existing files were tampered with: {tampered}")
    print("Leaderboard update aborted.")
    exit(1)

# Get only newly added files
new_submissions = [c['path'] for c in changes if c['status'] == 'A']

if not new_submissions:
    print("ℹ️ No new submissions found in this push. Recalculating leaderboard from existing files...")
    # Fallback to scanning folder if manually triggered or first run
    new_submissions = [os.path.join(r, f) for r, d, fs in os.walk("submissions") for f in fs if f.endswith(".enc")]

# If multiple new files, pick the latest one by git commit time
if len(new_submissions) > 1:
    print(f"⚠️ Multiple new files found: {new_submissions}. Selecting the most recent...")
    latest_file_cmd = 'git log --diff-filter=A --format="%ct %H" --name-only HEAD^..HEAD | awk \'NF==2 {t=$1; next} /\.enc$/ {print t, $0}\' | sort -nr | head -n 1'
    latest_output = subprocess.check_output(latest_file_cmd, shell=True).decode('utf-8')
    selected_file = latest_output.split()[-1]
    print(f"✅ Selected latest: {selected_file}")
    target_files = [selected_file]
else:
    target_files = new_submissions

# --- 3. PROCESSING ---
leaderboard_path = 'leaderboard/leaderboard.csv'
if os.path.exists(leaderboard_path):
    current_df = pd.read_csv(leaderboard_path)
else:
    # Added 'Last Updated' column here
    current_df = pd.DataFrame(columns=['Team', 'MAE', 'Last Updated'])

# Note: target_files is now a list of dictionaries from the new get_git_file_info()
for entry in target_files:
    file_path = entry['path']
    submission_time = entry['sub_time']  # This is the actual commit time
    team_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        with open(file_path, 'rb') as f:
            decrypted_csv = decrypt_file(f.read(), priv_key)
        
        if decrypted_csv:
            pred_df = pd.read_csv(StringIO(decrypted_csv))
            new_score = calculate_mae(gt_df, pred_df)
            
            if new_score is not None:
                # Update logic: Only keep if better (lower MAE) or new team
                if team_name in current_df['Team'].values:
                    old_score = current_df.loc[current_df['Team'] == team_name, 'MAE'].values[0]
                    
                    if new_score < old_score:
                        current_df.loc[current_df['Team'] == team_name, 'MAE'] = new_score
                        # Update timestamp to the time of this new personal best
                        current_df.loc[current_df['Team'] == team_name, 'Last Updated'] = submission_time
                        print(f"🔥 New Personal Best for {team_name}: {new_score:.8f}")
                    else:
                        print(f"keep: {team_name}'s new score ({new_score:.8f}) was not better than existing ({old_score:.8f})")
                else:
                    # New Entry: Store the score and the commit timestamp
                    new_row = pd.DataFrame({
                        'Team': [team_name], 
                        'MAE': [new_score], 
                        'Last Updated': [submission_time]
                    })
                    current_df = pd.concat([current_df, new_row], ignore_index=True)
                    print(f"✨ New Entry: {team_name} scored {new_score:.8f}")

    except Exception as e:
        print(f"❌ Failed {team_name}: {e}")

# --- 4. FINAL RANKING & EXPORT ---
if not current_df.empty:
    # 1. Clean and Prepare Data
    current_df['MAE'] = pd.to_numeric(current_df['MAE'], errors='coerce')
    current_df = current_df.dropna(subset=['MAE'])

    # 2. Sort and Rank
    # Primary Sort: MAE (Lowest is best)
    # Secondary Sort: Last Updated (Earliest submission wins the tie)
    current_df = current_df.sort_values(
        by=['MAE', 'Last Updated'], 
        ascending=[True, True]
    )
    
    # Calculate 'dense' rank based on MAE only (so identical scores share a rank)
    current_df['Rank'] = current_df['MAE'].round(8).rank(method='dense').astype(int)
    
    # 3. Final selection (including the new column)
    final_df = current_df[['Rank', 'Team', 'MAE', 'Last Updated']]
    
    os.makedirs('leaderboard', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # Save CSV with 8 decimals
    final_df.to_csv(leaderboard_path, index=False, float_format='%.8f')
    
    # Save Markdown
    with open('leaderboard/LEADERBOARD.md', 'w') as f:
        f.write("# 🏆 Competition Leaderboard\n\n" + final_df.to_markdown(index=False, floatfmt=".8f"))

    # HTML generation (Adjusted formatters for the new column layout)
    html_table = final_df.to_html(
        classes='table table-hover text-center', 
        index=False,
        formatters={'MAE': lambda x: f"{x:.8f}"}
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <title>OASIS3 Challenge</title>
        <style>
            body {{ background-color: #f4f7f6; font-family: 'Inter', sans-serif; padding: 40px 0; }}
            .leaderboard-card {{ background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; max-width: 900px; margin: auto; }}
            .header-section {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 40px 20px; }}
            table {{ width: 100% !important; margin-bottom: 0 !important; }}
            th {{ background-color: #f8fafc !important; color: #64748b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; text-align: center; padding: 15px !important; }}
            td {{ vertical-align: middle; font-size: 0.95rem; padding: 15px !important; text-align: center; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="leaderboard-card">
            <div class="header-section text-center">
                <h1 class="fw-bold">🧠 Brain-Age Prediction Leaderboard</h1>
                <div class="badge bg-primary mt-2">Last Updated: {pd.Timestamp.now().strftime('%b %d, %H:%M UTC')}</div>
            </div>
            <div class="table-responsive">
                {html_table}
            </div>
        </div>
    </body>
    </html>
    """
    with open('docs/leaderboard.html', 'w') as f:
        f.write(html_content) 

    print("🎉 Leaderboard files updated with time-based tie-breaking.")
else:
    print("❌ No valid scores to display.")