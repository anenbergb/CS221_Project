
import threading
from Queue import Queue

#Define a thread class to parse the avro file (filename)
#Thread executes tasks from a given tasks queue
class Worker(threading.Thread):
	def __init__(self, tasks):
		threading.Thread.__init__(self)
		#tasks is the shared task queue
		self.tasks = tasks
		self.daemon = True
		self.start()
	def run(self):
		while True:
		    func, args, kargs = self.tasks.get()
		    try:
		        func(*args, **kargs)
		    except Exception, e:
		        print e
		    finally:
		        self.tasks.task_done()

#ThreadPool consuming tasks form a queue
class ThreadPool:
	def __init__(self, num_threads):
		self.tasks = Queue(num_threads)
		for _ in range(num_threads): Worker(self.tasks)
	def add_task(self, func, *args, **kargs):
		#Add a task to the queue
		self.tasks.put((func, args, kargs))
	def wait_completion(self):
		#Wait for completion of all the tasks in the queue
		self.tasks.join()