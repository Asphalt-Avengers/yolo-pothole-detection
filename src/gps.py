import serial
import pynmea2

def get_gps_coordinates(port="/dev/ttyAMA0", baudrate=9600, timeout=0.5):
    """
    Reads a single valid GPS GPRMC sentence from the serial port and returns the latitude and longitude.

    Parameters:
    - port (str): The serial port to read from.
    - baudrate (int): Baud rate for the GPS module.
    - timeout (float): Serial timeout in seconds.

    Returns:
    - tuple: (latitude, longitude) if valid data is found, else (None, None)
    """
    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        dataout = pynmea2.NMEAStreamReader()

        while True:  
            newdata = ser.readline().decode("unicode_escape")
            
            if newdata[0:6] == "$GPRMC":  
                try:
                    newmsg = pynmea2.parse(newdata)
                    lat, lng = newmsg.latitude, newmsg.longitude
                    return lat, lng  
                except pynmea2.ParseError:
                    continue  

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None, None
    

