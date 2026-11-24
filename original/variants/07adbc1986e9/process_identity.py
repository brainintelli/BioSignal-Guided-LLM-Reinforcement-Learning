def is_worker(args,script,output):
    return len(args)>=5 and args[0].endswith(b'python3.12') and args[1:5]==[b'-u',script,b'collect',output]
