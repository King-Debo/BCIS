# Import the required libraries and modules
import sys

# Import the brain_lib, brain_ml, brain_aug, brain_enh, brain_com, brain_norm, brain_gui, and brain_db modules
import brain_lib as bl
import brain_ml as bm
import brain_aug as ba
import brain_enh as be
import brain_com as bc
import brain_norm as bn
import brain_gui as bg
import brain_db as bd

# Define the global variables and constants
EEG_DEVICE = 'eeg_device' # The name of the EEG device
FMRI_DEVICE = 'fmri_device' # The name of the fMRI device
OPTO_DEVICE = 'opto_device' # The name of the optogenetics device

# Define the main function to run the system
def main():
    # Create a brain GUI object
    gui = bg.BrainGUI()

    # Enter the main loop of the GUI
    gui.window.mainloop()

    # Close the brain GUI object
    gui.close()

# Run the main function if the script is executed
if __name__ == '__main__':
    main()
