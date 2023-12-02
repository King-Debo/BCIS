# Import the required libraries and modules
import numpy as np
import socket as sk
import threading as th

# Import the brain_lib, brain_ml, and brain_aug modules
import brain_lib as bl
import brain_ml as bm
import brain_aug as ba

# Define the global variables and constants
HOST = 'localhost' # The host address
PORT = 8080 # The port number
BUFFER_SIZE = 4096 # The buffer size
FORMAT = 'utf-8' # The encoding format
SEPARATOR = '<SEP>' # The separator symbol

# Define the class for communicating with the brain using EEG
class BrainSender:
    def __init__(self):
        # Initialize the brain sender
        self.augmentor = ba.EEGAugmentor() # Create an EEG augmentor object
        self.socket = sk.socket(sk.AF_INET, sk.SOCK_STREAM) # Create a socket object
        self.socket.connect((HOST, PORT)) # Connect to the host and port

    def send(self, data):
        # Send the EEG data to the host
        data = self.augmentor.enhance(data) # Enhance the EEG data
        data = data.tobytes() # Convert the data to bytes
        self.socket.send(data) # Send the data to the socket

    def close(self):
        # Close the brain sender
        self.augmentor.close()
        self.socket.close()

# Define the class for communicating with the brain using fMRI
class BrainReceiver:
    def __init__(self):
        # Initialize the brain receiver
        self.stimulator = ba.FMRIStimulator() # Create an fMRI stimulator object
        self.socket = sk.socket(sk.AF_INET, sk.SOCK_STREAM) # Create a socket object
        self.socket.bind((HOST, PORT)) # Bind to the host and port
        self.socket.listen() # Listen for incoming connections

    def receive(self):
        # Receive the fMRI data from the host
        conn, addr = self.socket.accept() # Accept a connection
        data = b'' # Initialize an empty byte string
        while True:
            chunk = conn.recv(BUFFER_SIZE) # Receive a chunk of data
            if not chunk: # If no more data, break the loop
                break
            data += chunk # Append the chunk to the data
        data = np.frombuffer(data, dtype=np.float64) # Convert the data from bytes to a numpy array
        data = self.stimulator.enhance(data) # Enhance the fMRI data
        return data

    def close(self):
        # Close the brain receiver
        self.stimulator.close()
        self.socket.close()

# Define the class for communicating with the brain using optogenetics
class BrainCommunicator:
    def __init__(self):
        # Initialize the brain communicator
        self.emulator = ba.OptoEmulator() # Create an optogenetics emulator object
        self.sender = BrainSender() # Create a brain sender object
        self.receiver = BrainReceiver() # Create a brain receiver object
        self.thread = th.Thread(target=self.communicate) # Create a thread object
        self.thread.start() # Start the thread

    def communicate(self):
        # Communicate with the brain using optogenetics
        while True:
            data = self.receiver.receive() # Receive the fMRI data from the host
            data = self.emulator.decode(data) # Decode the optogenetics data
            self.emulator.emulate(data) # Emulate the optogenetics data
            self.sender.send(data) # Send the EEG data to the host

    def close(self):
        # Close the brain communicator
        self.emulator.close()
        self.sender.close()
        self.receiver.close()
        self.thread.join() # Join the thread
