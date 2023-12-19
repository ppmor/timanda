from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class Report:
    def __init__(self, filename):
        self.canvas = canvas.Canvas(filename, pagesize=letter)
        self.width, self.height = letter
        self.current_height = self.height - 50

    def dodaj_tekst(self, tekst):
        if self.current_height < 40:
            self.canvas.showPage()
            self.current_height = self.height - 50

        self.canvas.drawString(50, self.current_height, tekst)
        self.current_height -= 20

    def dodaj_wykres(self, file_path, width=400, height=300):
        if self.current_height < height:
            self.canvas.showPage()
            self.current_height = 800

        self.canvas.drawImage(
            file_path,
            50,
            self.current_height - height,
            width=width,
            height=height
        )
        self.current_height -= (height + 50)

    def zapisz(self):
        self.canvas.save()
