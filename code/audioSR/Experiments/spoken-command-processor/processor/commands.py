import os

import model
import redis

from . import messaging, record, utils


class UserCommand(object):
    def __init__(self, username, message, dest_port, model_output):
        self.username = username
        self.message = message
        self.dest_port = dest_port
        self.model_output = model_output

    def save(self):
        """Save the command to the database (Redis)"""
        r = self.db_conn()

        key = '%s:%s' % (self.username, self.model_output)
        r.hmset(key, {'message':self.message, 'port':self.dest_port})

    @classmethod
    def find_closest_match(cls, username, model_output, n=1):
        """Pass speech_input through the model, get a prediction (a list of
        words) and output the guessed commands (words)

        Args:
            n: limits the number of returned matches

        Returns:
            message, dest_port
        """
        r = cls.db_conn()

        # format: "derek:*"
        raw_keys = r.keys('%s:*' % username)

        # without the usernames and the ":", the keys are the messages
        commands = [k[len(username) + 1:] for k in raw_keys]

        # get each command's edit distance with respect to model_output and
        # rank accordingly (start with the closest match)
        distances = [(c, utils.edit_distance(model_output, c)) for c in commands]
        distances.sort(key=lambda x: x[1])

        for command, distance in distances:
            key = '%s:%s' % (username, command)

            message = r.hmget(key, 'message')[0]
            port = r.hmget(key, 'port')[0]

            yield message, int(port)

    @classmethod
    def db_conn(cls):
        """Get Redis connection object to modify database accordingly"""
        return redis.StrictRedis(host='localhost', port=6379, db=0)

def register():
    """Go through the register process (ask for the fields in UserCommand) to
    save it to the database.
    """
    print('Registering command...')

    # Prompt for parameters
    username = raw_input('>>> username: ')
    message = raw_input('>>> message: ')
    dest_port = raw_input('>>> destination port: ')

    record_flag = raw_input('>>> press "y" to start recording: ')
    if record_flag == 'y':
        recording, wavfile = record.record_input(wavfile=os.environ['TMP_RECORDING'])

        mfccs = model.utils.wavfile_to_mfccs(wavfile)[0]
        model_output = model.predict(mfccs)[0]

        UserCommand(username, message, dest_port, model_output).save()

def parse():
    """Go through the process of parsing a user's speech (take his username
    + prompt to record), and then feed the recording to the model in order
    to get an output.
    """
    username = raw_input('>>> username: ')
    record_flag = raw_input('>>> press "y" to start recording: ')

    if record_flag == 'y':
        recording, wavfile = record.record_input(wavfile=os.environ['TMP_RECORDING'])

        mfccs = model.utils.wavfile_to_mfccs(wavfile)[0]
        model_output = model.predict(mfccs)

        matches = [m for m in UserCommand.find_closest_match(username, model_output, n=10)]
        print('>>> confirm your message:')

        for count, match in enumerate(matches):
            print('>>>\t%d:\tPORT: %d \tMESSAGE: %s' % (count, match[1], match[0]))

        chosen_input = int(raw_input('>>> choice: '))
        if chosen_input >= len(matches):
            print('>>> invalid choice; ending.\n')
            return

        message, port = matches[chosen_input]
        if messaging.send(message, port):
            print('>>> sending: PORT: %d, MESSAGE %s\n' % (port, message))
