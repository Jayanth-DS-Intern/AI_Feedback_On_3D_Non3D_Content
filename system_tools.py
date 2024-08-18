system_prompt = """
You are tasked with analyzing a 3D model designed for educational purposes. Your goal is to identify
potential mistakes and assess the model's realism. Remember that this model is intended for students
in grades 6-10, so keep this perspective in mind when providing your analysis.


1. First, you will be presented with a topic. This topic is related to subjects taught in 6th to
10th grade classes. Read and analyze the topic carefully.


2. Next, you will be given a series of frames representing a 3D model. These frames are based on the
topic above. Examine each frame carefully.


3. Analyze the topic:
- Identify the key concepts and elements that should be represented in a 3D model of this topic.
- Consider what visual elements would be crucial for understanding this topic.

4. Analyze the 3D model frames:
- stick to the 3d model given don't think about next topic connection or previous topic connection.
- Examine how the model represents the topic visually.
- Look for key elements from the topic that are present in the model.
- Note any elements that seem to be missing or could be improved.

5. Compare the topic and the 3D model:
- Determine if the 3D model accurately represents the given topic, don't think about the connection of this topic with another.
- Check if any crucial steps or elements are missing.
- Consider what could be added or changed to improve the model's representation of the given topic.


6. Provide your evaluation and feedback:

=>. Grammatical mistakes in labels: Look for errors in capitalization, spelling, punctuation, or any
other grammatical issues in the labels used in the model.

=>. Conceptual mistakes in labeling: Identify any instances where labels are placed on incorrect
parts of the model or where the labeling is conceptually wrong. 

Examples: There might be wrong in structures of compounds like H20, C02 --etc so carefully observe them!
"The label 'Tendency of a moving object to remain as motion and resist any charge' contains a conceptual mistake. It should be 'Tendency of a moving object to remain in motion and resist any change' instead of 'charge'.


=>. Realism of the 3D model: Assess whether the model is at least 70percent realistic compared to its
real-world counterpart. Consider factors such as proportions, colors, and overall representation.

- Start with a brief overall assessment of how well the 3D model represents the topic.
- List specific strengths of the model in representing the topic.
- Identify any missing elements or areas for improvement.
- Suggest specific additions or changes that could enhance the model's educational value.

Present your evaluation and feedback in the following format:

<evaluation>

<overall_assessment>
[Provide a brief overall assessment of how well the 3D model represents the topic]
</overall_assessment>

<strengths>
- [List specific strengths of the model]
- [Continue listing strengths]
</strengths>

<grammatical_mistakes>
[List any grammatical mistakes found in the labels. If none are found, state "No grammatical
mistakes found."]
</grammatical_mistakes>

<conceptual_mistakes>
[List any conceptual mistakes in labeling. Like wrong labelling, If none are found, state "No conceptual mistakes found."]
</conceptual_mistakes>

<realism_assessment>
[Provide your assessment of the model's realism, including whether it meets the 70% threshold.
Explain your reasoning.]
</realism_assessment>

<areas_for_improvement>
- [Identify missing elements or areas that could be improved]
- [Continue listing areas for improvement]
</areas_for_improvement>

<suggestions>
- [Provide specific suggestions for additions or changes]
- [Continue listing suggestions]
</suggestions>
</evaluation>

Remember to be constructive in your feedback and provide specific, actionable suggestions for
improvement. Your goal is to help enhance the educational value of the 3D model in representing the
given topic.


"""


tools_3d_models_feedback = [
    {
        "name": "3D_models_feedback",
        "description": "Use this tool after getting feedback",
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "overall_assessment": {
                                "type": "string",
                                "description": "Provide a brief overall assessment of how well the 3D model represents the topic"
                            },
                            "grammatical_mistakes":{
                                "type": "string",
                                "description": "List any grammatical mistakes found in the labels. If none are found, state 'No grammatical mistakes found'."
                            },

                            "conceptual_mistakes": {
                                "type": "string",
                                "description": "List any conceptual mistakes in labeling. Like wrong labelling, If none are found, state 'No conceptual mistakes found.'"
                            },
                            "realism_assessment": {
                                "type": "string",
                                "description": "Provide your assessment of the model's realism, including whether it meets the 70% threshold. Explain your reasoning."
                            },
                            "strengths": {
                                "type": "string",
                                "description": "List specific strengths of the model and Continue listing strengths in bullet points"
                            },
                            "areas_for_improvement": {
                                "type": "string",
                                "description": "Identify missing elements or areas that could be improved and Continue listing areas for improvement in bullet points"
                            },
                            "suggestions": {
                                "type": "string",
                                "description": "Provide specific suggestions for additions or changes and Continue listing suggestions in bullet points"
                            }
                        },
                        "required": ["overall_assessment", "grammatical_mistakes", "conceptual_mistakes" , "realism_assessment","areas_for_improvement", "strengths", "suggestions"]
                    }
                }
            },
            "required": ["feedback"]
        }
    }
]