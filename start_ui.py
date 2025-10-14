# start_ui.py
import os
import sys
import subprocess
import webbrowser
import threading
import time

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open("http://localhost:8501")

def main():
    print("🎯" * 50)
    print("📈 STOCK TREND PREDICTOR - AI POWERED")
    print("🎯" * 50)
    print("\n🚀 Launching application...")
    print("⏳ This may take a few seconds...")
    print("\n💡 TIPS:")
    print("   • Use sidebar to enter stock names")
    print("   • Try: RELIANCE, TCS, INFY, HDFC BANK")
    print("   • Adjust settings in the sidebar")
    print("   • View technical charts in main panel")
    print("\n" + "="*50)
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "stock_ui.py"])
    except KeyboardInterrupt:
        print("\n\n❌ Application stopped by user")
    except Exception as e:
        print(f"\n\n💥 Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure all packages are installed")
        print("   • Check if port 8501 is available")
        print("   • Try: python -m pip install streamlit")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()