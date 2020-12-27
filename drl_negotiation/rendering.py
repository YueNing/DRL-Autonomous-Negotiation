import pyglet
import time

class Viewer(pyglet.window.Window):

    def __init__(self, x, y, agent):
        super().__init__()
        self.agent = agent
        self.label = pyglet.text.Label(
                    agent.name,
                    font_name='Times New Roman',
                    font_size=36,
                    x=self.width//2, y=self.height//2,
                    anchor_x='center', anchor_y='center'
                )

    def on_draw(self):
        self.clear()
        self.label.draw()

    def _load_new_state(self):
        with open(f"{self.agent.name}.log", 'r') as fp:
            result = fp.read()

        return result
    
    def update(self):
        self.label.text = self._load_new_state() 

    def render(self, return_rgb_array=False):
        self.clear()
        self.switch_to()
        self.dispatch_events() 
        self.update()
        self.on_draw()
        
        arr = None
        if return_rgb_array:
            pass
        self.flip()
        return arr

if __name__ == '__main__':
    viewer01 = Viewer('00MyC@0')
    for _ in range(100):
        time.sleep(1)
        viewer01.render()
