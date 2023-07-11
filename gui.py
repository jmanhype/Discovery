import tkinter as tk

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Agent Simulation")
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.user_input = None
        self.entry.bind("<Return>", self.get_entry)

    def draw_agent(self, agent, x, y):
        self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=agent.color)
        self.canvas.create_text(x, y + 15, text=agent.id, font=("Arial", 8))

    def update(self, world, agents):
        self.canvas.delete("all")

        for agent in agents:
            x, y = world.get_agent_position(agent)
            self.draw_agent(agent, x, y)

        self.root.update()

    def get_entry(self, event):
        self.user_input = self.entry.get()
        self.entry.delete(0, tk.END)

    def get_user_input(self):
        temp = self.user_input
        self.user_input = None
        return temp

    def close_window(self):
        self.root.destroy()