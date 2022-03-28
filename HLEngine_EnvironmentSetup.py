#author:Akhil P Jacob
#HLDynamic-Integrations
import subprocess
import sys
import time
def setup_libraries():
    from xml.dom import minidom
    mydoc = minidom.parse('payload_setup.xml')
    payload = mydoc.getElementsByTagName('payload')    
    xfile=open("HL_Logs/log.txt","w")
    xfile.write("")
    xfile.close()    
    for elem in payload:
        print(elem.firstChild.data)
        package = str(elem.firstChild.data)
        try:
            print("HLEngine : Processing......")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except:
            print("HLEngine: Installation failed....")
            xfile=open("log.txt","a")
            xfile.write(package)
            xfile.close()


try:
    print("\nWelcome to HL_ENGINE Development Platform 2020 - Robot Development Simplified")
    time.sleep(1)
    print("\nDesigned and Developed by: Er.Akhil P Jacob (last updated on 24th March 2020)")
    time.sleep(1)
    print("\nThe Setup will take time depending on the internet speed and the system performance.")
    print("\nIt is recommended to close all other applications including Editors or IDE's on installation")
    print("\n processing..........")
    
    time.sleep(5)
    print("\nHLEngine Robot Developmental Environment setup console initializing.....")
    time.sleep(2)
    setup_libraries()
except:
    
    print("HLEngine: failed to commence. Please install manually")

