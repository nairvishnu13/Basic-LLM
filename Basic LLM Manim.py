from manim import *
import random

# Global Settings
FONT = "Segoe UI"
TEXT_SIZE = 22
TITLE_SIZE = 28

class LLM(ThreeDScene):
    def construct(self):
        # Timeline of scenes
        self.p1_motivation()
        self.p2_character_model()
        self.p3_training_data()
        self.p4_vocab_and_tokenization()
        self.p5_input_target_mapping()
        self.p6_transformer_pipeline_overview()
        self.p7_training_big_picture()
        self.p7_token_and_position_embeddings()
        self.p8_transformer_block()
        self.p9_output_loss_learning()
        self.p10_results()
        self.p11_conclusion()
        

    def p1_motivation(self):
        t1 = Text("In the age of Generative AI, you might wonder..", font=FONT, font_size=TITLE_SIZE)
        t2 = Text("how does it actually work?", font=FONT, font_size=TITLE_SIZE)
        group = VGroup(t1, t2).arrange(DOWN, buff=0.3)

        # P1 Thinking emoji (bottom-left)
        thinking = ImageMobject("Manim/emoji/thinking_f.png")
        thinking.scale(0.3)
        thinking.to_corner(DR, buff=0.5)

        self.play(Write(group), FadeIn(thinking, scale=0.6), run_time=3, lag_ratio=0.3 )
        self.wait(1)

        # P1 3. Subtle thinking motion
        self.play(
            thinking.animate.shift(UP * 0.15),
            rate_func=there_and_back,
            run_time=1.2
        )

        self.wait(3)


        self.play(FadeOut(group),FadeOut(thinking))

    def p2_character_model(self):
        t = Text("Let's explore this using a simple example - a character-level model", 
                 font=FONT, font_size=TITLE_SIZE, line_spacing=1.5)
        sub = Text("Predicting the next character from a tiny sequence", font_size=28, color=BLUE, font=FONT).next_to(t, DOWN, buff=1)
        self.play(Write(t),run_time=2)
        self.wait(1)
        self.play(FadeIn(sub, shift=UP),run_time=2)
        self.wait(3)
        self.clear()

    def p3_training_data(self):
        title = Text("Training Data", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        text = Text("Consider a very simple training dataset:", font_size=24,font=FONT).to_edge(UP, buff=2.5)
        data = Text("abcd abcd abcd abcd abcd...", font="Consolas",font_size=24, color=BLUE).to_edge(UP,buff=3.5)
        goal = Text("Goal: predict the next character, without any input to the model", font_size=24, font=FONT,color=YELLOW).to_edge(DOWN, buff=2)
        self.add(title)
        self.wait(1)
        self.play(Write(text), run_time=2)
        self.wait(2)
        self.play(Write(data), run_time=2)
        self.wait(2)
        self.play(Write(goal), run_time=2)
        self.wait(3)
        self.clear()

    def p4_vocab_and_tokenization(self):
        title = Text("Vocabulary & Tokenization", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        
        text1 = MarkupText("Let’s create a vocabulary from our training data",font=FONT, font_size=TEXT_SIZE)
        text2 = MarkupText("The characters are: a, b, c, d, ' ' with vocab_size 5", font=FONT, font_size=TEXT_SIZE)
        text3 = MarkupText("Now we'll assign integers to these characters \nwhich will be used for indexing", font=FONT, font_size=TEXT_SIZE)
        text4 = MarkupText("We can think of the text as a sequence of integers:", font=FONT, font_size=TEXT_SIZE)
        text5 = MarkupText("1, 2, 3, 4, 0, 1, 2, 3, 4, 0..", font=FONT, font_size=TEXT_SIZE)
        text6 = MarkupText("So, our goal becomes to train the model \nand predict the next integer", font=FONT, font_size=TEXT_SIZE,color=YELLOW)


        title1 = VGroup(text1,text2,text3).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.3
        )
        title1.to_edge(LEFT, buff=.4)
        title1.to_edge(UP, buff=1.4)

        title2 = VGroup(text4,text5).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.3
        )
        title2.to_edge(LEFT, buff=7.5)
        title2.to_edge(UP, buff=2)
        text6.to_edge(LEFT, buff=7.5)
        text6.to_edge(UP, buff=5)

        matrix_values1 = r"\begin{bmatrix} \text{' '} \\ a \\ b \\ c \\ d \end{bmatrix}"
        matrix_values2 = r"\begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \\ 4 \end{bmatrix}"
        #matrix_values4_3 = r"\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}"

        # Create MathTex object
        matrix1 = MathTex(matrix_values1, font_size=40)
        matrix2 = MathTex(matrix_values2, font_size=40)

        # Center it
        matrix1.to_edge(LEFT,buff=2.2)
        matrix1.to_edge(UP,buff=4)
        matrix2.to_edge(LEFT,buff=4.4)
        matrix2.to_edge(UP,buff=4)

        arrow = Arrow(
            matrix1.get_right(),
            matrix2.get_left(),
            buff=0.4
        )
        map_text = Text("Vocabulary Size: 5", font_size=18, font=FONT, color=YELLOW).next_to(arrow, DOWN, buff=1.4)

        separator = Line(
            start=UP * 2.7,
            end=DOWN * 2.9,
            stroke_width=2,
            color=GREY_B
        )

        separator.set_x(0).shift(RIGHT*.1)  # center vertically on the screen
        

        # ============================================================
        # P4 Add to scene
        self.add(title)
        self.play(Write(title1), run_time=10)
        self.wait(3)
        self.add(matrix1)
        self.wait(1)
        self.add(arrow)
        self.wait(1)
        self.add(matrix2)
        self.wait(1)
        self.add(map_text)
        self.wait(1)
        self.play(FadeIn(separator))
        self.wait(1)
        self.play(Write(title2), run_time=7)
        self.wait(3)
        self.play(Write(text6), run_time=5)
        self.wait(4)
        
        

        self.wait(1)
        self.clear()

    def p5_input_target_mapping(self):
        title = Text("Input (X) vs Target (Y)", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        text1 = MarkupText("- Training uses input–target pairs, where the target is the input shifted by one token",font=FONT, font_size=TEXT_SIZE)
        text2 = MarkupText("- The model learns next-token prediction by mapping input tokens to their next tokens",font=FONT, font_size=TEXT_SIZE)
        

        textgroup1 = VGroup(text1,text2).arrange(DOWN,aligned_edge=LEFT,buff=0.3)
        textgroup1.to_edge(UP, buff=1.4)
        
        matrix_values1 = r"\begin{bmatrix} \text{' '} \\ a \\ b \\ c \\ d \end{bmatrix}"
        matrix_values2 = r"\begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \\ 4 \end{bmatrix}"
        matrix_values3 = r"\begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 0 \end{bmatrix}"
        matrix_values4 = r"\begin{bmatrix} a \\ b \\ c \\ d \\ \text{' '} \end{bmatrix}"

        # Create MathTex object
        matrix1 = MathTex(matrix_values1, font_size=40)
        matrix2 = MathTex(matrix_values2, font_size=40)
        matrix3 = MathTex(matrix_values3, font_size=40)
        matrix4 = MathTex(matrix_values4, font_size=40)
         

        group1 = VGroup(matrix1, matrix2).arrange(RIGHT, buff=1.5)
        group1.to_edge(LEFT, buff=2.2).to_edge(UP, buff=4)

        arrow1 = Arrow(matrix1.get_right(),matrix2.get_left(),buff=0.4)
        group1 = VGroup(matrix1, arrow1, matrix2)

        box1 = SurroundingRectangle(group1,
            buff=0.3,        # space between objects and box
            corner_radius=0.1,
            color=BLUE
        )

        group2 = VGroup(matrix3, matrix4).arrange(RIGHT, buff=1.5)
        group2.to_edge(LEFT, buff=9).to_edge(UP, buff=4)

        arrow2 = Arrow(matrix3.get_right(),matrix4.get_left(),buff=0.4)
        group2 = VGroup(matrix3, arrow2, matrix4)

        box1 = SurroundingRectangle(group1,
            buff=0.3,        # space between objects and box
            corner_radius=0.1,
            color=BLUE
        )

        box2 = SurroundingRectangle(group2,
            buff=0.3,        # space between objects and box
            corner_radius=0.1,
            color=BLUE
        )

        arrow3 = Arrow(box1.get_right(),box2.get_left(),buff=0.4)

        map_text1 = Text("Input", font_size=18, font=FONT, color=YELLOW).next_to(box1, UP, buff=0.2)
        map_text2 = Text("Transformation", font_size=18, font=FONT, color=YELLOW).next_to(arrow3, UP, buff=0.2)
        map_text3 = Text("Target", font_size=18, font=FONT, color=YELLOW).next_to(box2, UP, buff=0.2)




        self.add(title)
        self.play(Write(textgroup1), run_time=10)
        
        
        self.play(FadeIn(group1))
        self.wait(2)
        self.play(Create(box1),run_time=1)
        self.wait(1)
        self.play(Write(map_text1))
        self.wait(2)
        self.play(Write(arrow3),Write(map_text2), run_time=3)
        self.wait(2)
        self.play(FadeIn(group2))
        self.wait(2)
        self.play(Create(box2),run_time=1)
        self.wait(1)
        self.play(Write(map_text3))
        self.wait(2)
        self.wait(3)

        
        self.clear()

    def p6_transformer_pipeline_overview(self):
        title = Text("The Transformation Pipeline", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        BOX_WIDTH = 2.5
        BOX_HEIGHT = 0.5
        
        def make_box(label):
            box = RoundedRectangle(
                width=BOX_WIDTH,
                height=BOX_HEIGHT,
                corner_radius=0.08
            )
            text = Text(label, font=FONT, font_size=18)
            text.move_to(box)
            return VGroup(box, text)

        # -------------------------------------------------
        # Flowchart boxes
        # -------------------------------------------------
        boxes = VGroup(
            make_box("Tokens"),
            make_box("Embeddings"),
            make_box("Transformer Blocks"),
            make_box("Logits"),
            make_box("Probabilities"),
            make_box("Loss")
        ).arrange(DOWN, buff=0.45)

        boxes.to_edge(LEFT, buff=0.1).to_edge(UP, buff=1.4)

        # -------------------------------------------------
        # Arrows
        # -------------------------------------------------
        arrows = VGroup(
            *[
                Arrow(
                    boxes[i][0].get_bottom(),
                    boxes[i + 1][0].get_top(),
                    buff=0.1
                )
                for i in range(len(boxes) - 1)
            ]
        )

        # -------------------------------------------------
        # One-line explanations (right side)
        # -------------------------------------------------
        explanations = VGroup(
            Text("Discrete numbers representing characters/words",
                 font=FONT, font_size=18),

            Text("Each token mapped to a learnable vector",
                 font=FONT, font_size=18),

            Text("Tokens exchange information using attention",
                 font=FONT, font_size=18),

            Text("Raw scores for each possible next token",
                 font=FONT, font_size=18),

            Text("Scores normalized into probabilities",
                 font=FONT, font_size=18),

            Text("Single number measuring prediction error",
                 font=FONT, font_size=18),
        )

        for expl, box in zip(explanations, boxes):
            expl.next_to(box, RIGHT, buff=0.1)
            expl.set_y(box.get_center()[1])

        # -------------------------------------------------
        # Math explanation
        # -------------------------------------------------
        math_title = Text(
            "Core computation inside the model",
            font=FONT,
            font_size=18,
            color=YELLOW
        )

        math_eq = MathTex(
            r"Y = W X + b",
            font_size=22
        )

        math_eq_expanded = MathTex(
            r"Y_i = \sum_j W_{ij} X_j + b_i",
            font_size=22
        )

        math_text = VGroup(
            Text("X  → input activations (token vectors)",
                 font=FONT, font_size=18),
            Text("W  → learned weights (what the model stores)",
                 font=FONT, font_size=18),
            Text("b  → bias (learned offset)",
                 font=FONT, font_size=18),
            Text("Y  → transformed output activations",
                 font=FONT, font_size=18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        math_group = VGroup(
            math_title,
            math_eq,
            math_eq_expanded,
            math_text
        ).arrange(DOWN, buff=0.4)

        math_group.to_edge(RIGHT, buff=0.1).to_edge(UP, buff=1.4)
        #math_group.align_to(explanations, LEFT)

        # -------------------------------------------------
        # Learning explanation
        # -------------------------------------------------
        learning_text = Text(
            "Training adjusts W and b\n"
            "to reduce the loss over time.",
            font=FONT,
            font_size=18, color=YELLOW
        ).next_to(math_group, DOWN, buff=0.6)

        separator = Line(start=boxes.get_right() + RIGHT*2 + UP*boxes.height/2,
                 end=boxes.get_right() + RIGHT*2 + DOWN*boxes.height/2,
                 stroke_width=2, stroke_opacity=0.4)

        separator.next_to(explanations, RIGHT, buff=0.25)

        # -------------------------------------------------
        # Animations
        # -------------------------------------------------
        self.play(Write(title))
        self.wait(0.5)

        for box, arrow, expl in zip(boxes[:-1], arrows, explanations[:-1]):
            self.play(FadeIn(box),run_time=1.5)
            self.play(Write(expl),run_time=3.5)
            self.play(GrowArrow(arrow))
            

        self.play(FadeIn(boxes[-1]))
        self.play(FadeIn(explanations[-1]))
        
        self.play(FadeIn(separator))
        self.wait(1)
        self.play(FadeIn(math_group),runtime=3)
        self.wait(5)
        self.play(Write(learning_text),runtime=3)

        self.wait(7)
        self.clear()

    def p7_training_big_picture(self):
        title = Text("Training - Big Picture", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        text1 = MarkupText("- Training proceeds over many batches sampled from the dataset",font=FONT, font_size=TEXT_SIZE)
        text2 = MarkupText("- Each batch contains multiple token sequences processed in parallel",font=FONT, font_size=TEXT_SIZE)
        text3 = MarkupText("- A batch with sequence of 8 tokens and batch size 4 (parallel runs) looks like:",font=FONT, font_size=TEXT_SIZE)
        text4 = MarkupText("- Tokens are mapped to embedding vectors before entering the Transformer",font=FONT, font_size=TEXT_SIZE)
        text5 = MarkupText("- The Transformer operates on continuous embeddings, not token IDs",font=FONT, font_size=TEXT_SIZE)
        text6 = MarkupText("- Logits are generated via unembedding and converted to probabilities",font=FONT, font_size=TEXT_SIZE)
        text7 = MarkupText("- Average batch loss is computed and parameters are updated via backpropagation",font=FONT, font_size=TEXT_SIZE)
        text8 = MarkupText("For sample calculations we'll \nlook at first row this batch",font=FONT, font_size=18,color=YELLOW)

        textgroup1 = VGroup(text1,text2,text3).arrange(DOWN,aligned_edge=LEFT,buff=0.3)
        textgroup1.to_edge(UP, buff=1.4)

        textgroup2 = VGroup(text4,text5,text6,text7).arrange(DOWN,aligned_edge=LEFT,buff=0.3)
        textgroup2.to_edge(DOWN, buff=0.4)
        
        matrix_values1 = r"\begin{bmatrix} 4 & 0 & 1 & 2 & 3 & 4 & 0 & 1 \\ 1 & 2 & 3 & 4 & 0 & 1 & 2 & 3 \\ 2 & 3 & 4 & 0 & 1 & 2 & 3 & 4 \\ 0 & 1 & 2 & 3 & 4 & 0 & 1 & 2 \end{bmatrix}"
        

        # Create MathTex object
        matrix1 = MathTex(matrix_values1, font_size=40).shift(DOWN * 0.2) 
        
        text8.next_to(matrix1,RIGHT, buff=0.3)

        box1 = Rectangle(height=.5,width=5.5,
            
            color=YELLOW
        ).shift(UP * 0.60) 


        
        #arrows = VGroup(*[Arrow(flow[i], flow[i+1]) for i in range(len(flow)-1)])
        self.add(title)
        self.play(Write(textgroup1), run_time=10)
        self.wait(4)
        self.add(matrix1)
        self.wait(3)
        self.play(Write(textgroup2), run_time=15)

        self.play(Create(box1),run_time=2)
        self.play(Write(text8), run_time=3)
        self.wait(5)
        self.clear()

    def p7_token_and_position_embeddings(self):
        title = Text("Token Embeddings", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        text1 = MarkupText("- Each token is mapped to a continuous vector (embedding size = 4 for below case)",font=FONT, font_size=TEXT_SIZE)
        text2 = MarkupText("- Position embeddings are added so the model knows where each token appears",font=FONT, font_size=TEXT_SIZE)
        
        textgroup1 = VGroup(text1,text2).arrange(DOWN,aligned_edge=LEFT,buff=0.3)
        textgroup1.to_edge(UP, buff=1.4)

        matrix_values1 = r"\begin{bmatrix} 4 & 0 & 1 & 2 & 3 & 4 & 0 & 1 \\ 1 & 2 & 3 & 4 & 0 & 1 & 2 & 3 \\ 2 & 3 & 4 & 0 & 1 & 2 & 3 & 4 \\ 0 & 1 & 2 & 3 & 4 & 0 & 1 & 2 \end{bmatrix}"
        matrix_values2 = r"\begin{bmatrix} 4 \\ 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ 0 \\ 1 \end{bmatrix}"
        matrix_values3 = r"\begin{bmatrix} -0.56 & 0.26 & 0.75 & -0.51\\ 1.99 & -0.55 & -1.13 & -1.96\\ -1.06 & 0.21 & -0.44 & 2.51\\ 0.07 & -0.34 & -3.7 & -1.5\\ -1.15 & -1.32 & 2.61 & -0.46\\ -0.22 & 0.11 & 0.56 & -0.12\\ -0.52 & -0.53 & -1.12 & -1.07\\ 1.53 & -0.13 & -0.95 & -0.53 \end{bmatrix}"

        # Create MathTex object
        matrix1 = MathTex(matrix_values1, font_size=40).shift(DOWN * 0.2)
        matrix2 = MathTex(matrix_values2, font_size=40).to_edge(LEFT,buff=2).shift(DOWN * 0.5)
        matrix3 = MathTex(matrix_values3, font_size=40).shift(RIGHT * 2).shift(DOWN * 0.5)

        box1 = Rectangle(height=.5,width=5.5,
            
            color=YELLOW
        ).shift(UP * 0.60) 

        objectgroup1 = VGroup(matrix1,box1)

        arrow1 = Arrow(matrix2.get_right(),matrix3.get_left(),buff=0.4)

        # # Placeholder matrix for token vectors
        # tok_emb = Matrix([["v_{1,1}", "v_{1,2}"], ["v_{2,1}", "v_{2,2}"], ["v_{3,1}", "v_{3,2}"]]).scale(0.6).shift(LEFT*3)
        # pos_emb = Matrix([["p_{1,1}", "p_{1,2}"], ["p_{2,1}", "p_{2,2}"], ["p_{3,1}", "p_{3,2}"]]).scale(0.6).shift(RIGHT*3)
        # plus = MathTex("+").move_to(ORIGIN)
        
        self.add(title)
        self.add(objectgroup1)
        self.wait(2)
        self.play(ReplacementTransform(objectgroup1, matrix2), run_time=5)
        self.play(Write(textgroup1), run_time=8)
        self.wait(2)
        
        self.play(Write(arrow1), run_time=2)
        self.wait(1)
        self.play(FadeIn(matrix3),run_time=1)
        self.wait(3)
        self.clear()

    def p8_transformer_block(self):
        title = Text("Transformer Block", font=FONT, font_size=TITLE_SIZE).to_edge(UP)
        title1 = Text("Normalization", font=FONT, font_size=TEXT_SIZE, color=YELLOW)
        text1 = MarkupText("- Normalize activations to keep values stable",font=FONT, font_size=18)
        
        equation1 = MathTex(
            r"\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}",
            font_size=40, color=BLUE
        ).shift(DOWN * 2.6).shift(LEFT * 4)
        text1_1 = MarkupText(" Re-scales activations to have a mean of 0 and a variance of 1",font=FONT, font_size=20,color=BLUE).next_to(equation1,RIGHT*1)

        title2 = Text("Multi-Head Attention", font=FONT, font_size=TEXT_SIZE, color=YELLOW)
        text2 = MarkupText("- Tokens interact with each other using Queries, Keys, and Values",font=FONT, font_size=18)
        text3 = MarkupText("- Multiple heads learn different relationships",font=FONT, font_size=18)
        #text4 = MarkupText("- Attention uses learned projections:",font=FONT, font_size=18)
        
        text3_1 = Text("Splits embeddings into multiple heads so the model can focus on different relationships \n(syntax, semantics, long-range context) in parallel",font=FONT, font_size=20,color=BLUE).set_alignment("center").shift(DOWN * 2.5).shift(LEFT * 0)
        text3_2 = Text("Intuition: In “The cat sat on the couch, and it slept,” one head links “it → cat” (coreference), \nwhile another links “cat → slept” (who did the action)",font=FONT, font_size=20,color=BLUE).set_alignment("center").next_to(text3_1,DOWN*1).shift(RIGHT * .5)

        # CAT emoji (bottom-left)
        cat = ImageMobject("Manim/emoji/cat_f.png")
        cat.scale(0.3)
        cat.to_corner(DL, buff=0.3)

        # ============================================================



        title3 = Text("Feed-Forward Network", font=FONT, font_size=TEXT_SIZE, color=YELLOW)
        text5 = MarkupText("- Each token updates itself using scaling + non-linearity",font=FONT, font_size=18)

        text5_1 = Text("After tokens share information via attention, the FFN independently refines each token’s representation, \nturning 'what I know' into 'what I should become'",font=FONT, font_size=20,color=BLUE).set_alignment("center").shift(DOWN * 2.5).shift(LEFT * 0)
        
        
        title4 = Text("Normalization + Attention + Feed-Forward", font=FONT, font_size=TEXT_SIZE, color=YELLOW)
        text6 = MarkupText("- Forms one Transformer block",font=FONT, font_size=18)
        text7 = MarkupText("- This block is repeated multiple times",font=FONT, font_size=18)

        text7_1 = Text("Residual connection: The outputs of Normalization, Attention, and Feed-Forward layers are added back to the \noriginal activations, helping preserve information and stabilize training",font=FONT, font_size=20,color=BLUE).set_alignment("center").shift(DOWN * 2.5).shift(LEFT * 0)
        text7_2 = Text("Dropout: To reduce overfitting, activations are randomly set to zero during training, encouraging the model \nto learn robust features",font=FONT, font_size=20,color=BLUE).set_alignment("center").next_to(text7_1,DOWN*1)


        textgroup1 = VGroup(title1,text1).arrange(DOWN,aligned_edge=LEFT,buff=0.15).to_edge(RIGHT,buff=2.4).to_edge(UP,buff=1.4)
        textgroup2 = VGroup(title2,text2,text3).arrange(DOWN,aligned_edge=LEFT,buff=0.15).next_to(textgroup1,DOWN*1.5).align_to(textgroup1, LEFT)
        textgroup3 = VGroup(title3,text5).arrange(DOWN,aligned_edge=LEFT,buff=0.15).next_to(textgroup2,DOWN*1.5).align_to(textgroup1, LEFT)
        textgroup4 = VGroup(title4,text6,text7).arrange(DOWN,aligned_edge=LEFT,buff=0.15).next_to(textgroup3,DOWN*1.5).align_to(textgroup1, LEFT)


        matrix_values1 = r"\begin{bmatrix} 4 \\ 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ 0 \\ 1 \end{bmatrix}"
        matrix_values2 = r"\begin{bmatrix} -0.56 & 0.26 & 0.75 & -0.51\\ 1.99 & -0.55 & -1.13 & -1.96\\ -1.06 & 0.21 & -0.44 & 2.51\\ 0.07 & -0.34 & -3.7 & -1.5\\ -1.15 & -1.32 & 2.61 & -0.46\\ -0.22 & 0.11 & 0.56 & -0.12\\ -0.52 & -0.53 & -1.12 & -1.07\\ 1.53 & -0.13 & -0.95 & -0.53 \end{bmatrix}"
        matrix_values3 = r"\begin{bmatrix} -0.99 & 0.5 & 1.39 & -0.91\\ 1.63 & -0.09 & -0.49 & -1.05\\ -1.01 & -0.07 & -0.55 & 1.63\\ 0.98 & 0.7 & -1.59 & -0.09\\ -0.67 & -0.78 & 1.7 & -0.24\\ -1.02 & 0.1 & 1.58 & -0.66\\ 1 & 0.99 & -1.09 & -0.91\\ 1.65 & -0.11 & -0.99 & -0.55 \end{bmatrix}"
        matrix_values4 = r"\begin{bmatrix} -0.76 & -0.08 & 0.82 & -0.85\\ 1.46 & -1.34 & -0.63 & -2.38\\ -1.49 & -0.29 & -0.44 & 2.57\\ -0.38 & -0.96 & -3.36 & -1.78\\ -1.59 & -1.82 & 2.56 & -0.33\\ -0.62 & -0.33 & 0.51 & -0.06\\ -0.91 & -1.05 & -0.8 & -1.41\\ 1.08 & -0.75 & -0.55 & -0.88 \end{bmatrix}"
        matrix_values5 = r"\begin{bmatrix} -1.07 & -0.63 & 0.14 & -0.91\\ 1.45 & -1.75 & -1.03 & -2.11\\ -1.25 & -0.63 & -0.6 & 2.84\\ 0.02 & -1.14 & -3.42 & -1.58\\ -1.77 & -2.34 & 1.97 & -0.56\\ -0.73 & -0.87 & -0.04 & -0.22\\ -1.18 & -1.55 & -1.5 & -1.24\\ 1.18 & -1.17 & -0.87 & -0.57 \end{bmatrix}"
        matrix_values6 = r"\begin{bmatrix} -0.64 & -3.68 & -0.61 & -1.81\\ 2.31 & -2.26 & -1.51 & -2.91\\ -0.2 & -2.79 & -0.3 & 1.45\\ 1.7 & -1.67 & -3.38 & -2.51\\ -1.32 & -4.44 & 1.7 & -1.63\\ -0.25 & -3.36 & -0.08 & -1.33\\ -0.84 & -2.95 & -1.56 & -2.51\\ 2.1 & -1.69 & -1.07 & -1.58 \end{bmatrix}"


        matrix1 = MathTex(matrix_values1, font_size=40).to_edge(LEFT,buff=2).shift(DOWN * 0.5)
        matrix2 = MathTex(matrix_values2, font_size=40).shift(RIGHT * 2).shift(DOWN * 0.5)
        
        arrow1 = Arrow(matrix1.get_right(),matrix2.get_left(),buff=0.4)
        objectgroup1 = VGroup(matrix1,matrix2,arrow1)

        ## use this as initial

        matrix3 = MathTex(matrix_values1, font_size=35).to_edge(LEFT,buff=.1).shift(DOWN * 0)
        matrix4 = MathTex(matrix_values2, font_size=35).next_to(matrix3,RIGHT*1.2).shift(DOWN * 0)
        matrix5 = MathTex(matrix_values3, font_size=35).next_to(matrix3,RIGHT*1.2).shift(DOWN * 0)
        matrix6 = MathTex(matrix_values4, font_size=35).next_to(matrix3,RIGHT*1.2).shift(DOWN * 0)
        matrix7 = MathTex(matrix_values5, font_size=35).next_to(matrix3,RIGHT*1.2).shift(DOWN * 0)
        matrix8 = MathTex(matrix_values6, font_size=35).next_to(matrix3,RIGHT*1.2).shift(DOWN * 0)
        
        arrow2 = Arrow(matrix3.get_right(),matrix4.get_left(),buff=0.4)
        objectgroup2 = VGroup(matrix3,matrix4,arrow2)


        p1 = np.array([-1.1, 1.6, 0])
        p2 = np.array([-1.1, 1.15, 0])
        p3 = np.array([-1.1, 0.7, 0])
        p4 = np.array([-1.1, 0.25, 0])
        p5 = np.array([-1.1, -.2, 0])
        p6 = np.array([-1.1, -.65, 0])
        p7 = np.array([-1.1, -1.1, 0])
        p8 = np.array([-1.1, -1.5, 0])


        # 2. Create the curved path using ArcBetweenPoints
        # Adjust 'radius' to make it more or less curved
        curve1 = ArcBetweenPoints(p1, p2, radius=-.25, color=RED)
        curve2 = ArcBetweenPoints(p1, p3, radius=-.5, color=RED)
        curve3 = ArcBetweenPoints(p1, p4, radius=-.75, color=RED)
        curve4 = ArcBetweenPoints(p1, p5, radius=-1.2, color=RED)
        curve5 = ArcBetweenPoints(p1, p6, radius=-1.5, color=RED)
        curve6 = ArcBetweenPoints(p1, p7, radius=-1.8, color=RED)
        curve7 = ArcBetweenPoints(p1, p8, radius=-2.1, color=RED)
        
        # 3. Add a tip to make it an arrow
        # arrow = Arrow(color=RED).add_tip() 
        # # We use TipableVMobject properties to turn the arc into an arrow
        # curve1.add_tip(tip_length=0.1)

        



        self.add(title)
        self.add(objectgroup1)
        self.play(ReplacementTransform(objectgroup1, objectgroup2), run_time=5)
        self.play(Write(textgroup1), run_time=5)

        self.play(FadeOut(matrix4), run_time=1.5)
        self.play(FadeIn(matrix5), run_time=1.5)
        self.wait(1)
        
        self.play(Write(equation1), run_time=2)
        self.play(Write(text1_1), run_time=2)
        
        self.wait(4)
        self.play(FadeOut(equation1),FadeOut(text1_1), run_time=1.5)
        
        self.play(Write(textgroup2), run_time=8)
        
        
        self.play(Create(curve1),Create(curve2),Create(curve3),Create(curve4),Create(curve5),Create(curve6),Create(curve7),run_time=2)

        self.play(FadeOut(matrix5), run_time=1.5)
        self.play(FadeIn(matrix6), run_time=1.5)
        self.wait(1)

        self.play(Write(text3_1), run_time=5)
        self.play(Write(text3_2), run_time=5)

        # CAT
        self.play(
        FadeIn(cat, scale=0.6),
        run_time=3,
        lag_ratio=0.3 )
        self.wait(1)

        # P1 3. Subtle thinking motion
        self.play(
            cat.animate.shift(UP * 0.15),
            rate_func=there_and_back,
            run_time=1.2
        )
        self.wait(1)   

        self.wait(4)
        self.play(FadeOut(curve1),FadeOut(curve2),FadeOut(curve3),FadeOut(curve3),FadeOut(curve4),FadeOut(curve5),FadeOut(curve6),FadeOut(curve7), run_time=1.5)
        self.play(FadeOut(text3_1),FadeOut(text3_2),FadeOut(cat), run_time=1.5)
        self.wait(1) 
        self.play(Write(textgroup3), run_time=7)

        self.play(FadeOut(matrix6), run_time=1.5)
        self.play(FadeIn(matrix7), run_time=1.5)
        self.wait(1)

        self.play(Write(text5_1), run_time=6)
        self.wait(4)
        self.play(FadeOut(text5_1), run_time=1.5)

        self.play(Write(textgroup4), run_time=8)

        self.play(FadeOut(matrix7), run_time=1.5)
        self.play(FadeIn(matrix8), run_time=1.5)
        self.wait(1)

        self.play(Write(text7_1), run_time=7)
        self.play(Write(text7_2), run_time=7)

        self.wait(2)
        

        self.wait(3)
        self.clear()

    
    def p9_output_loss_learning(self):
        title = Text("Output, Loss and Learning", font=FONT, font_size=TITLE_SIZE).to_edge(UP)

        text1 = Text("- After the final block, activations are normalized again", font=FONT, font_size=20).to_edge(RIGHT,buff=.02).to_edge(UP,buff=1.4)
        text2 = Text("- We multiply activations with the unembedding matrix \nto convert vectors → logits", font=FONT, font_size=20).next_to(text1,DOWN*1.5).align_to(text1, LEFT)
        text3 = Text("- Logits → probabilities: We compare predictions with \ntargets and compute the loss", font=FONT, font_size=20).next_to(text2,DOWN*1.5).align_to(text1, LEFT)
        text4 = Text("- The loss acts like a penalty", font=FONT, font_size=20).next_to(text3,DOWN*1.5).align_to(text1, LEFT)
        text5 = Text("- Backpropagation: We compute gradients and slightly \nadjust weights and biases to reduce future loss", font=FONT, font_size=20).next_to(text4,DOWN*1.5).align_to(text1, LEFT)
        text6 = Text("- After training, these weights are used to generate \nnew characters based on the last input sequence", font=FONT, font_size=20).next_to(text5,DOWN*1.5).align_to(text1, LEFT)
        text7 = Text("Target", font=FONT, font_size=20,color=GREEN)
        text8 = Text("Since the model is untrained, it assigns higher scores to tokens 3 and 4 instead of the true target 2", font=FONT, font_size=20,color=BLUE).set_alignment("center").shift(DOWN * 2.5).shift(LEFT * 0)
        text9 = Text("During training, the loss captures this mismatch, and backpropagation computes gradients that \nupdate the weights - boosting the correct token’s score and suppressing the others over time", font=FONT, font_size=20,color=BLUE).set_alignment("center").shift(DOWN * 2.5).shift(LEFT * 0)

        

        #textgroup1 = VGroup(text1).arrange(DOWN,aligned_edge=LEFT,buff=0.15).to_edge(RIGHT,buff=2.4).to_edge(UP,buff=1.4)
        #textgroup2 = VGroup(text2).arrange(DOWN,aligned_edge=LEFT,buff=0.15).next_to(textgroup1,DOWN*1).align_to(textgroup1, LEFT)
        
        
        matrix_values1 = r"\begin{bmatrix} 4 \\ 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ 0 \\ 1 \end{bmatrix}"
        matrix_values2 = r"\begin{bmatrix} -0.64 & -3.68 & -0.61 & -1.81\\ 2.31 & -2.26 & -1.51 & -2.91\\ -0.2 & -2.79 & -0.3 & 1.45\\ 1.7 & -1.67 & -3.38 & -2.51\\ -1.32 & -4.44 & 1.7 & -1.63\\ -0.25 & -3.36 & -0.08 & -1.33\\ -0.84 & -2.95 & -1.56 & -2.51\\ 2.1 & -1.69 & -1.07 & -1.58 \end{bmatrix}"
        matrix_values3 = r"\begin{bmatrix} 0.84 & -1.6 & 0.86 & -0.1\\ 1.68 & -0.58 & -0.21 & -0.9\\ 0.17 & -1.54 & 0.1 & 1.26\\ 1.64 & -0.11 & -0.99 & -0.54\\ 0.05 & -1.39 & 1.44 & -0.09\\ 0.77 & -1.61 & 0.9 & -0.06\\ 1.37 & -1.2 & 0.49 & -0.66\\ 1.71 & -0.73 & -0.33 & -0.66  \end{bmatrix}"
        matrix_values4 = r"\begin{bmatrix} -0.45 & 0.32 & 0.53 & 1.54 & 0.4\\ -0.07 & -0.64 & 0.29 & 1.24 & 1.12\\ -0.09 & 0.51 & 0.33 & 0.38 & 0.63\\ 0.26 & -0.94 & 0.08 & 0.52 & 1.49\\ -0.67 & 0.45 & 0.36 & 1.39 & -0.21\\ -0.46 & 0.35 & 0.52 & 1.52 & 0.36\\ -0.33 & -0.13 & 0.48 & 1.63 & 0.72\\ -0.01 & -0.56 & 0.33 & 1.18 & 1.23  \end{bmatrix}"
        matrix_values5 = r"\begin{bmatrix} 0.64 & 1.38 & 1.7 & 4.68 & 1.5\\ 0.93 & 0.53 & 1.33 & 3.47 & 3.07\\ 0.92 & 1.67 & 1.39 & 1.46 & 1.87\\ 1.29 & 0.39 & 1.08 & 1.68 & 4.46\\ 0.51 & 1.57 & 1.44 & 4.01 & 0.81\\ 0.63 & 1.42 & 1.69 & 4.59 & 1.44\\ 0.72 & 0.88 & 1.61 & 5.09 & 2.06\\ 0.99 & 0.57 & 1.39 & 3.25 & 3.41  \end{bmatrix}"
        matrix_values6 = r"\begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ 0 \\ 1 \\ 2 \end{bmatrix}"
        

        matrix1 = MathTex(matrix_values1, font_size=35).to_edge(LEFT,buff=.1).shift(DOWN * 0)
        matrix2 = MathTex(matrix_values2, font_size=35).next_to(matrix1,RIGHT*1.2).shift(DOWN * 0)
        matrix3 = MathTex(matrix_values3, font_size=35).next_to(matrix1,RIGHT*1.2).shift(DOWN * 0)
        matrix4 = MathTex(matrix_values4, font_size=35).next_to(matrix1,RIGHT*1.2).shift(DOWN * 0)
        matrix5 = MathTex(matrix_values5, font_size=35).next_to(matrix1,RIGHT*1.2).shift(DOWN * 0)
        matrix6 = MathTex(matrix_values6, font_size=35,color=GREEN).next_to(matrix5,RIGHT*1.2).shift(DOWN * 0)

        text7.next_to(matrix6,UP*1.2)

        p1 = np.array([-5.8, 3, 0])
        p2 = np.array([-4.8,3, 0])
        p3 = np.array([-3.8, 3, 0])
        p4 = np.array([-2.8, 3, 0])
        p5 = np.array([-1.8, 3, 0])

        n1=MathTex(str(0), font_size=35, color=YELLOW).next_to(p1,DOWN*1)
        n2=MathTex(str(1), font_size=35, color=YELLOW).next_to(p2,DOWN*1)
        n3=MathTex(str(2), font_size=35, color=YELLOW).next_to(p3,DOWN*1)
        n4=MathTex(str(3), font_size=35, color=YELLOW).next_to(p4,DOWN*1)
        n5=MathTex(str(4), font_size=35, color=YELLOW).next_to(p5,DOWN*1)
    
        
        # d1=Dot(p1)
        # d2=Dot(p2)
        # d3=Dot(p3)
        # d4=Dot(p4)
        # d5=Dot(p5)
        
        arrow1 = Arrow(matrix1.get_right(),matrix2.get_left(),buff=0.4)
        objectgroup1 = VGroup(matrix1,matrix2,arrow1)
        objectgroup2 = VGroup(matrix6,text7)

        a1 = Arrow(n1.get_bottom(),n1.get_bottom() + DOWN * 0.7, buff=0.1, color=YELLOW)
        a2 = Arrow(n2.get_bottom(),n2.get_bottom() + DOWN * 0.7, buff=0.1, color=YELLOW)
        a3 = Arrow(n3.get_bottom(),n3.get_bottom() + DOWN * 0.7, buff=0.1, color=YELLOW)
        a4 = Arrow(n4.get_bottom(),n4.get_bottom() + DOWN * 0.7, buff=0.1, color=YELLOW)
        a5 = Arrow(n5.get_bottom(),n5.get_bottom() + DOWN * 0.7, buff=0.1, color=YELLOW)

        rect = Rectangle(width=2.7,height=.55,color=BLUE).shift(DOWN*1.53).shift(LEFT*1.9)
        

        self.add(title)
        self.add(objectgroup1)

        self.play(Write(text1), run_time=4)

        self.play(FadeOut(matrix2), run_time=1.5)
        self.play(FadeIn(matrix3), run_time=1.5)
        self.wait(1)


        self.play(Write(text2), run_time=5)
        
        self.play(FadeOut(matrix3), run_time=1.5)
        self.play(FadeIn(matrix4), run_time=1.5)
        self.wait(1)


        self.play(Write(text3), run_time=5)

        self.play(FadeOut(matrix4), run_time=1.5)
        self.play(FadeIn(matrix5), run_time=1.5)
        self.wait(1)

        # self.play(FadeIn(d1), FadeIn(d2),FadeIn(d3), FadeIn(d4),FadeIn(d5) ,run_time=1.5)
        self.play(FadeIn(n1), FadeIn(n2),FadeIn(n3), FadeIn(n4),FadeIn(n5) ,run_time=2)
        self.play(FadeIn(a1), FadeIn(a2),FadeIn(a3), FadeIn(a4),FadeIn(a5) ,run_time=2)
        self.wait(1)

        self.play(FadeIn(matrix6),FadeIn(text7), run_time=2)
        self.wait(1)
        
        self.play(FadeIn(rect), run_time=2)
        self.wait(2)

        self.play(Write(text8), run_time=7)
        self.wait(3)


        self.play(Write(text4), run_time=3)
        self.wait(1)
        
        self.play(FadeOut(text8), run_time=1.5)
        self.wait(1)
        
        self.play(Write(text5), run_time=8)
        self.wait(1)
        self.play(Write(text9), run_time=8)
        
        self.wait(1)
        self.play(Write(text6), run_time=5)

        self.wait(6)
        self.clear()


    def p10_results(self):
        text1 = Text("Let's look at the results once the model has learned:", font=FONT, font_size=TITLE_SIZE, line_spacing=1.5).shift(UP*1.7)
        text2 = Text("Input: abcd abcd abcd abcd abcd...", font_size=28, color=BLUE, font=FONT).next_to(text1, DOWN, buff=.8)
        text3 = Text("Output: abcd abcd abcd abcd abcd...", font_size=28, color=BLUE, font=FONT).next_to(text2, DOWN, buff=.8).align_to(text2, LEFT)

        text4 = Text("The results from the same model when trained on Harry Potter books:", font=FONT, font_size=TITLE_SIZE, line_spacing=1.5).shift(UP*1.7)
        text5 = Text("“Well,” said the blew, shiving passled. \n\n “‘Weak, he said,” said Hermione, hurried. \n\n “Place saved in the castle green sight, Malfoy's leg.”", font_size=28, color=BLUE, font=FONT).next_to(text4, DOWN, buff=.5)   
        

        self.play(Write(text1),run_time=3)
        self.wait(2)
        self.play(FadeIn(text2),FadeIn(text3),run_time=3)
        self.wait(5)
        self.play(FadeOut(text1),FadeOut(text2),FadeOut(text3),run_time=1.5)

        self.play(Write(text4),run_time=3)
        self.wait(2)
        self.play(FadeIn(text5),run_time=3)
        self.wait(5)
        self.clear()

    def p11_conclusion(self):
        text1 = Text("As expected, the output is not yet meaningful, mainly because:", font=FONT, font_size=24, line_spacing=1.5).shift(UP*2.1)
        text2 = Text("- It is a character-level model, not a word-level one", font_size=24, color=BLUE, font=FONT).next_to(text1, DOWN, buff=.2)
        text3 = Text("- It was trained on a very small dataset", font_size=24, color=BLUE, font=FONT).next_to(text2, DOWN, buff=.2).align_to(text2, LEFT)
        text4 = Text("Despite that, the model has clearly learned English-like structure, punctuation, and dialogue patterns.", font_size=22, font=FONT).next_to(text3, DOWN, buff=0.6).move_to(ORIGIN, aligned_edge=UP)

        text5 = Text("Large Language Models are just this idea scaled up with more data, more parameters, and more compute.", font=FONT, font_size=22, line_spacing=1, color=YELLOW).next_to(text4, DOWN, buff=0.8)
              

        self.play(Write(text1),run_time=3)
        self.wait(2)
        self.play(Write(text2),run_time=2)
        self.play(Write(text3),run_time=2)
        self.wait(3)
        self.play(Write(text4),run_time=5)
        self.wait(2)
        self.play(FadeIn(text5),run_time=3)
        self.wait(6)
        self.clear()

    
    #############################################################################
    