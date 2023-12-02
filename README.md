# BCIS: Blockchain-based Certificate Issuance System

BCIS is a web application that allows users to create, issue, and verify digital certificates using blockchain technology. The project is written in Python using the Flask framework and the web3 library. The project also uses Solidity to create smart contracts that run on the Ethereum network.

## Project Requirements

To run the project, you will need the following:

- Python 3.8 or higher
- Flask 2.0.1 or higher
- web3 5.23.1 or higher
- Ganache 2.5.4 or higher
- MetaMask browser extension

## Installation and Setup

To install and set up the project, follow these steps:

1. Clone the GitHub repository to your local machine:

```bash
git clone https://github.com/King-Debo/BCIS.git

Navigate to the project directory and install the required Python packages:
cd BCIS
pip install -r requirements.txt

Launch Ganache and create a new workspace. Import the accounts from the accounts.json file in the project directory.

Connect MetaMask to the Ganache network by selecting Custom RPC and entering the following details:

Network Name: Ganache
New RPC URL: http://127.0.0.1:7545
Chain ID: 1337
Import one of the Ganache accounts to MetaMask by using the private key.

Compile and deploy the smart contract Certificate.sol using Remix IDE or Truffle. Copy the contract address and paste it in the app.py file.

Run the Flask app by executing the following command:

python app.py

Open your browser and go to http://127.0.0.1:5000 to access the web application.
Usage Overview
The web application has three main features: create, issue, and verify certificates.

To create a certificate, you need to fill in the certificate details, such as the name, course, date, and issuer. You also need to upload a signature image and a logo image. The app will generate a QR code that contains the certificate hash and the contract address. You can download the certificate as a PDF file.

To issue a certificate, you need to enter the recipient’s email address and the certificate hash. The app will send an email to the recipient with a link to view and download the certificate. You also need to confirm the transaction on MetaMask to store the certificate hash on the blockchain.

To verify a certificate, you need to scan the QR code on the certificate or enter the certificate hash and the contract address manually. The app will check if the certificate hash matches the one stored on the blockchain and display the verification result.
