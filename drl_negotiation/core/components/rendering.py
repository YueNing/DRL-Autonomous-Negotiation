import pyglet
import time


class Viewer(pyglet.window.Window):

    def __init__(self, width, height, agent):
        super().__init__(width=width, height=height)
        self.agent = agent
        self.batch = pyglet.graphics.Batch()

        self.document_seller = pyglet.text.decode_text(agent if isinstance(agent, str) else agent.name)
        self.document_buyer = pyglet.text.decode_text(agent if isinstance(agent, str) else agent.name)
        self.document_seller.set_style(0, 0, dict(font_name='Arial', font_size=15, color=(255, 255, 255, 255)))
        self.document_buyer.set_style(0, 0, dict(font_name='Arial', font_size=15, color=(255, 255, 255, 255)))
        self.layout_seller = pyglet.text.layout.ScrollableTextLayout(document=self.document_seller,
                                                width=self.width / 2,
                                                height=self.height,
                                                batch=self.batch,
                                                multiline=True,
                                                wrap_lines=True)
        self.layout_seller.x = 0
        self.layout_seller.y = 0

        self.layout_buyer =  pyglet.text.layout.ScrollableTextLayout(document=self.document_buyer,
                                                width=self.width / 2,
                                                height=self.height,
                                                batch=self.batch,
                                                multiline=True,
                                                wrap_lines=True)
        self.layout_buyer.x = self.width / 2
        self.layout_buyer.y = 0

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.layout_seller.view_y = -self.layout_seller.content_height

    def _load_new_state(self):
        with open(f"{self.agent if isinstance(self.agent, str) else self.agent.name}.log", 'r') as fp:
            result = fp.readlines()

        return ''.join(result[-9:-5]), ''.join(result[-4:])
    
    def update(self):
        self.document_seller.text, self.document_buyer.text = self._load_new_state()


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
    viewer01 = Viewer(700, 700, '00MyC@0')
    while True:
        time.sleep(1)
        viewer01.render()
