
class Box:
    def area(self):
        return self.width * self.height
    def volume(self):
        return self.area() * self.depth

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self. depth = depth

# Create an instance of Box.
x = Box(10, 2, 5)

# Print area.
print(x.area())
print(x.volume())