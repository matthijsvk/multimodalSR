import os
from contextlib import closing
import socket
import sys


def send(message, port, host='localhost'):
    """Open a connection between PORT and destination_port, and send the
    message.
    """
    # Create a TCP/IP socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        destination_addr = (host, port)
        try: # Connect and send message
            s.connect(destination_addr)
            s.sendall(message)
        except socket.error as err:
            print('Error Code : %s, Message %s' % (str(err[0]), err[1]))

    return True

def listen(port, host='localhost'):
    """Listen on PORT for new connections on a continuous basis. Accept them
    and print their messages.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            # Bind socket to address
            s.bind((host, port))

            # Start listening
            backlog = 10
            s.listen(backlog)

            # Block to accept connection
            recv_bytes = 1024
            while True:
                conn, addr = s.accept()
                message = conn.recv(recv_bytes)

                # Print message
                print('Message (%s:%s): %s' % (addr[0], addr[1], message))

                conn.close()

        except socket.error as err:
            print('Error Code : %d, Message %s' % (err[0], err[1]))
            sys.exit()


if __name__ == '__main__':
    listen()
