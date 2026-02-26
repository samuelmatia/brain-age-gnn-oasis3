import os
import argparse
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad

def encrypt_file(input_file, public_key_path, output_file):
    # 1. Read the predictions
    with open(input_file, 'rb') as f:
        data = f.read()

    # 2. Load the Public RSA Key
    with open(public_key_path, 'r') as f:
        recipient_key = RSA.import_key(f.read())

    # 3. Generate a random AES session key (for the data)
    session_key = os.urandom(16)

    # 4. Encrypt the data with AES
    cipher_aes = AES.new(session_key, AES.MODE_CBC)
    ct_bytes = cipher_aes.encrypt(pad(data, AES.block_size))
    iv = cipher_aes.iv

    # 5. Encrypt the AES session key with the RSA Public Key
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    enc_session_key = cipher_rsa.encrypt(session_key)

    # 6. Combine everything into the output file
    # Format: [Length of Enc Key (2 bytes)][Enc Session Key][IV (16 bytes)][Encrypted Data]
    with open(output_file, 'wb') as f:
        f.write(len(enc_session_key).to_bytes(2, byteorder='big'))
        f.write(enc_session_key)
        f.write(iv)
        f.write(ct_bytes)
    
    print(f"✅ Success! Encrypted file saved as: {output_file}")
    print("You can now submit this .enc file via Pull Request.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt predictions for secure submission.")
    parser.add_argument("--input", required=True, help="Path to your predictions.csv")
    parser.add_argument("--key", default="keys/public_key.pem", help="Path to public_key.pem")
    parser.add_argument("--output", required=True, help="Path to save the .enc file (e.g., submissions/team_name/predictions.enc)")
    
    args = parser.parse_args()
    encrypt_file(args.input, args.key, args.output)