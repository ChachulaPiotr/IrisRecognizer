from PIL import Image, ImageDraw, ImageFont


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


class ImageMaker:
    def __init__(self, nh):
        self.size = 100
        self.nsize = 4
        self.nh = nh
        self.hiddencord = makeMatrix(nh, 2, 0.0)
        self.fontsize = self.size*2
        self.colorstrength = 1500
        for i in range(nh):
            self.hiddencord[i][0] = self.nh * self.size * 10/2
            self.hiddencord[i][1] = (i + 1) * self.size*self.nsize*2
        self.inputcord = makeMatrix(4, 2, 0.0)
        self.outputcord = makeMatrix(1, 2, 0.0)
        for i in range(4):
            self.inputcord[i][0] = self.nh * self.size * 10/10
            self.inputcord[i][1] = self.hiddencord[0][1]/2 + self.size*self.nsize *(i+1) * 2
        self.outputcord[0][0] = self.nh * self.size * 10 * 4/5
        self.outputcord[0][1] = round(nh * self.size * 10 / 2)
        self.biasicord = makeMatrix(1, 2, 0.0)
        self.biasocord = makeMatrix(1, 2, 0.0)
        self.biasicord[0][0] = self.nh * self.size * 10/5
        self.biasicord[0][1] = self.nh * self.size * 10+10
        self.biasocord[0][0] = round(nh * self.size * 10 * 3/5)
        self.biasocord[0][1] = self.nh * self.size * 10 + 10

    def makeImage(self, input, doutput, poutput, ri, ro, k):
        im = Image.new('RGB', (self.nh * self.size * 10+10, self.nh * self.size * 10+10), color='white')
        draw = ImageDraw.Draw(im)
        r = 0
        g = 0
        for i in range(self.nh):
            if (ro[0][i]>0):
                g = round(ro[0][i]*self.colorstrength)
            else:
                r = -round(ro[0][i]*self.colorstrength)
            draw.ellipse([(self.hiddencord[i][0] - self.size * self.nsize/ 2, self.hiddencord[i][1] - self.size*self.nsize / 2),
                          (self.hiddencord[i][0] + self.size * self.nsize/ 2, self.hiddencord[i][1] + self.size * self.nsize / 2)],
                         (r, g, 0))
            r = 0
            g = 0
        if (poutput-doutput > 0):
            g = round((poutput-doutput) * self.colorstrength)
        else:
            r = -round((poutput-doutput) * self.colorstrength)
        draw.ellipse([(self.outputcord[0][0] - self.size * self.nsize/ 2, self.outputcord[0][1] - self.size*self.nsize / 2),
                          (self.outputcord[0][0] + self.size * self.nsize/ 2, self.outputcord[0][1] + self.size * self.nsize / 2)],
                         (r, g, 0))
        r = 0
        g = 0
        font = ImageFont.truetype("arial.ttf", self.fontsize)
        for i in range(4):
            draw.text((self.inputcord[i][0]-self.nsize*2, self.inputcord[i][1]+self.nsize*2), str(input[i]), (125, 125, 125), font)

        for i in range(4):
            for j in range(self.nh):
                if (ri[j][i] > 0):
                    g = round(ri[j][i] * self.colorstrength)
                else:
                    r = -round(ri[j][i] * self.colorstrength)
                draw.line([(self.inputcord[i][0]+round(self.size)*2, self.inputcord[i][1]),
                           (self.hiddencord[j][0], self.hiddencord[j][1])],
                          (r, g, 0), 20)
                r = 0
                g = 0
        for j in range(self.nh):
            if (ri[j][4] > 0):
                g = round(ri[j][4] * self.colorstrength)
            else:
                r = -round(ri[j][4] * self.colorstrength)
            draw.line([(self.biasicord[0][0] + round(self.size), self.biasicord[0][1]),
                       (self.hiddencord[j][0], self.hiddencord[j][1])],
                      (r, g, 0), 20)
            r = 0
            g = 0
        for j in range(self.nh):
            if (ro[0][j] > 0):
                g = round(ro[0][j] * self.colorstrength)
            else:
                r = -round(ro[0][j] * self.colorstrength)
            draw.line([(self.hiddencord[j][0] + round(self.size), self.hiddencord[j][1]),
                        (self.outputcord[0][0], self.outputcord[0][1])],
                        (r, g, 0), 20)
            r = 0
            g = 0
        if (ro[0][self.nh] > 0):
            g = round(ro[0][self.nh] * self.colorstrength)
        else:
            r = -round(ro[0][self.nh] * self.colorstrength)
        draw.line([(self.biasocord[0][0] + round(self.size), self.biasocord[0][1]),
                   (self.outputcord[0][0], self.outputcord[0][1])],
                  (r, g, 0), 20)
        draw.text((self.outputcord[0][0] + self.nsize * 2, self.outputcord[0][1] + self.nsize * 10), str(round(poutput,2))+ " : "+str(doutput),
                  (125, 125, 125), font)

        im.save(str(k)+".png")