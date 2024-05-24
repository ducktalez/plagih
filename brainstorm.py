from plagih.util import blue_string, BColors

x = """<?xml version="1.0" encoding="utf-8"?>
<EIX-LIST xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <ITEMS>
                               <ELEMENT xsi:type="string">&lt;padding&gt;</ELEMENT>
                               <ELEMENT xsi:type="string">&lt;padding&gt;</ELEMENT>
                               <ELEMENT format-rev="1" id="ed1ed4c6-ffbf-4d9f-9263-e94301c0301c" xsi:type="noEventTestStepMappingContainer">
                                               <TESTSTEP format-rev="3" id="1fe4cfad-088a-47b0-9ef3-3d1dbb8438e5" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Precondition</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP format-rev="3" id="67f401e4-910b-46e6-ab03-56e068e5a6f9" xsi:type="tsPackage">
                                                                               <PACKAGE-REFERENCE xsi:type="expressionref">
                                                                                              <PATH-EXPRESSION format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="string">C:/HIL_AE/Packages/Packages/FuSi/EATR/../../lib/HiL Init.pkg</VALUE>
                                                                                              </PATH-EXPRESSION>
                                                                               </PACKAGE-REFERENCE>
                                                                               <PARAM-ASSIGNMENTS/>
                                                                               <ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="94da3a57-84c2-4ef5-83c8-9b3358e909b9" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Ausgangszustand pruefen (FwgSigIn1_Vmin, FwgSigIn1_VMax lesen)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="1275d4a7-68f6-4616-b331-b9b38e5f2ab4" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="timelessOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9dd6715c-2aa9-49a8-9727-835619406fc4" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator1MaxLimit_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">accelerator1MaxLimit</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <EXPECTATION xsi:type="timelessOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">4.5</VALUE>
                                                                                                              </VALUE>
                                                                                                              <TOLERANCE style="percentage" xsi:type="Tolerance">
                                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                                                              </VALUE>
                                                                                                              </TOLERANCE>
                                                                                              </EXPRESSION>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_v</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="0d3602d6-0eda-4aae-b32b-49fe698b532c" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="6371bc6e-6069-4861-8dd7-d320c32ab577" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">accelerator2MaxLimit</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_v</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="727ecd69-d3d1-4abc-86b6-f610108b2997" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">maxGradient</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_undefined_perc/s</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="38c035c9-495c-4bfb-8c5a-1f990f077b3a" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">fwg2_v</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="d417c502-af59-42bc-9502-5cef1f9cd89a" name="TsStartTrace" xsi:type="utility-9308d99e-50ac-11dc-8fe5-001143176a1c">
                                                                               <NAME xsi:type="string">Reaktionszeit</NAME>
                                                                               <RECORDING-GROUP-REF-BY-UUID>6170fd80a38411eb9ac4c8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="7aeae68e-36f6-452b-8f27-8a601e34d37f" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">50</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="f21d1dea-a276-46a8-b3a2-641a2a5911f3" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern  (FwgSigIn1 &gt; FwgSigIn1_VMax setzen)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="87dcb11c-34b3-4527-9ce0-0ab2b9a97c13" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="11e1e13a-6a57-46bd-a76e-6d5b89aba6cf" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.2</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                               </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="4c407482-c988-4422-98e7-2b074b0902cd" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion pruefen (Safe state: &quot;invalid&quot; signal FwgSig1_v)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="510327a2-6114-40df-82f2-89462d4877ba" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">1.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">500</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="3115e0a0-d4fa-4d3d-93ed-69a7ef4aae37" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">pctAccRef</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9d3e4a26-d77e-4626-a99f-5e8aadf0f10e" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VARIABLE-REFS>
                                                                                              <VARIABLE-NAME dkey="default">
                                                                                                              <DVALUE xsi:type="string">fwg2_v</DVALUE>
                                                                                              </VARIABLE-NAME>
                                                                               </VARIABLE-REFS>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="08a18a7c-7f39-4dbf-9f98-12bf8cdc92aa" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">50</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="7779f48b-8953-4e34-b5e8-c16dbd328824" name="TsStopTrace" xsi:type="utility-d12f5791-50ac-11dc-8eb4-001143176a1c">
                                                                               <NAME xsi:type="string">Reaktionszeit</NAME>
                                                                               <RECORDING-GROUP-REF-BY-UUID>6170fd80a38411eb9ac4c8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="6f6d3849-61e5-4416-bf00-0bc5ad8afba8" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern (FWG2 manipulieren)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="94be66aa-0240-44b8-a214-6c36bdf40e0f" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="20233ec1-3386-4bca-8320-b8af55351bac" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">fwg2_v</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.2</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="07391d6e-c9aa-49ca-ac70-8f870b4760f3" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion pruefen (PctAcceleration folgt FWG2)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="7f3aa7ca-8b89-40b1-a545-b13f728667ca" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">==</RELATION>
                                                                                                              <VALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">1.0</VALUE>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">500</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="f2921e7f-8d22-4446-9e02-0312e78d2e88" xsi:type="tsRead">
                                                                               <MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
                                                                               <EXPECTATION xsi:type="finallyTrueOption">
                                                                                              <EXPRESSION xsi:type="builtNumericExpression">
                                                                                                              <RELATION xsi:type="string">&gt;=</RELATION>
                                                                                                              <VALUE xsi:type="binaryOpBaseExpression">
                                                                                                                              <NAME xsi:type="string">BINARY_ADD</NAME>
                                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                                              <NAME xsi:type="string">pctAccRef</NAME>
                                                                                                                              </FIRST-COMPONENT>
                                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                                              <VALUE xsi:type="integer">2</VALUE>
                                                                                                                              </SECOND-COMPONENT>
                                                                                                              </VALUE>
                                                                                              </EXPRESSION>
                                                                                              <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">1000</VALUE>
                                                                                              </TIME>
                                                                               </EXPECTATION>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <Z-UNIT xsi:type="string">u_none</Z-UNIT>
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="c9578c6b-4c6f-46ec-8dff-1b93acf1edfe" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zustand aendern (FWG2 Max Wert zur Gradientenbestimmung)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                               <TESTSTEP id="ff968496-8dcd-41e6-bf42-57af6cedffa8" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA xsi:type="binaryOpBaseExpression">
                                                                                                              <NAME xsi:type="string">BINARY_SUBTRACT</NAME>
                                                                                                              <FIRST-COMPONENT xsi:type="varBaseExpression">
                                                                                                                              <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                                                                              </FIRST-COMPONENT>
                                                                                                              <SECOND-COMPONENT format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="float">0.1</VALUE>
                                                                                                              </SECOND-COMPONENT>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="c21ce869-3253-4af6-b067-066e79b9f1fa" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Reaktion prüfen (Gradientenmessung per Traceanalyse)</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                               <TESTSTEP id="5c459f74-0dde-45e7-8de9-c783783c16f8" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
                                                                               <ENABLED xsi:type="boolean">False</ENABLED>
                                                                               <TIME format-rev="2" xsi:type="valueBaseExpression">
                                                                                              <VALUE xsi:type="integer">1000</VALUE>
                                                                               </TIME>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="735fe1b1-7860-4c13-9c6b-db86d1d85793" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Zuruecksetzen</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP id="93ab0edd-28a0-4f19-a8cb-4bdc198e4adc" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">0</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="babae8a8-576f-40a7-b2f1-d3ad0367fd28" xsi:type="tsRestore">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="9b075b78-d6a1-4a15-8b32-b047838afa6b" xsi:type="tsWrite">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
                                                                               <VALUE xsi:type="expressionValue">
                                                                                              <DATA format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="integer">0</VALUE>
                                                                                              </DATA>
                                                                               </VALUE>
                                                                               <METRIC format-rev="1" xsi:type="metricInfo">
                                                                                              <VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
                                                                                              <DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
                                                                               </METRIC>
                                                               </TESTSTEP>
                                                               <TESTSTEP id="a4404e8f-e2ed-463f-930d-dafcd2cc5b92" xsi:type="tsRestore">
                                                                               <MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <TESTSTEP format-rev="3" id="2d70af9b-6c3e-4071-a963-b6a9327938ad" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
                                                               <ACTION xsi:type="I18NItem">
                                                                               <MULTILANGDATA>
                                                                                              <ELEMENT dkey="de_DE">
                                                                                                              <DVALUE xsi:type="string">Postcondition</DVALUE>
                                                                                              </ELEMENT>
                                                                               </MULTILANGDATA>
                                                                               <INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
                                                               </ACTION>
                                                               <TESTSTEP format-rev="3" id="750af70a-b500-4b25-a1ee-2fa83fac9ec1" xsi:type="tsPackage">
                                                                               <PACKAGE-REFERENCE xsi:type="expressionref">
                                                                                              <PATH-EXPRESSION format-rev="2" xsi:type="valueBaseExpression">
                                                                                                              <VALUE xsi:type="string">C:\\HIL_AE\\Packages\\Packages\\lib\\TerminateExecution.pkg</VALUE>
                                                                                              </PATH-EXPRESSION>
                                                                               </PACKAGE-REFERENCE>
                                                                               <PARAM-ASSIGNMENTS>
                                                                                              <ASSIGNMENT dkey="Fahrzeug_stillstand">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Fehlerspeicher">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Klemmenwechsel_KL15">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">False</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                                              <ASSIGNMENT dkey="Klemmenwechsel_KL30">
                                                                                                              <DVALUE format-rev="2" xsi:type="valueBaseExpression">
                                                                                                                              <VALUE xsi:type="string">True</VALUE>
                                                                                                              </DVALUE>
                                                                                              </ASSIGNMENT>
                                                                               </PARAM-ASSIGNMENTS>
                                                                               <ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
                                                               </TESTSTEP>
                                               </TESTSTEP>
                                               <MAPPING format-rev="1" xsi:type="temporaryMappingSpace">
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveAcceleratorReceiver.UAccelerator1</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.PctAccelerator</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</ID>
                                                                               <XACCESS xsi:type="xaMeasValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</LABEL>
                                                                                              <RASTER xsi:type="string">Port-Default</RASTER>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_PctAcceleratorFailPositiveGradient_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_UAccelerator1MaxLimit_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_UAccelerator1MaxLimit_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</ID>
                                                                               <XACCESS xsi:type="xaCalibValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
                                                                                              <LABEL xsi:type="string">Sda_UAccelerator2MaxLimit_P</LABEL>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Active/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Control/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Active/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                                               <MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
                                                                               <ID xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</ID>
                                                                               <XACCESS format-rev="2" xsi:type="xaModelValueVariable">
                                                                                              <MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
                                                                                              <MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
                                                                                              <VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Control/Value</VARIABLE-PATH>
                                                                                              <VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
                                                                               </XACCESS>
                                                               </MAPPING-ITEM>
                                               </MAPPING>
                               </ELEMENT>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">DME1_DDE1_CPS_SafetyElectricDriveAcceleratorReceiver_UAccelerator1</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">maxGradient</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">maxGradient</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator1MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">fwg2_v</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">pctAccRef</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list">
                                               <ELEMENT format-rev="1" xsi:type="variable">
                                                               <NAME xsi:type="string">accelerator2MaxLimit</NAME>
                                                               <DEFAULT-VALUE xsi:type="undefined"/>
                                               </ELEMENT>
                               </ELEMENT>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list"/>
                               <ELEMENT xsi:type="list"/>
                </ITEMS>
</EIX-LIST>
"""
y = """<?xml version="1.0" encoding="utf-8"?>
<PACKAGE format-rev="7" prog-version="2020.2.98502" xmlns="http://www.tracetronic.de/xml/ecu-test" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.tracetronic.de/xml/ecu-test package.xsd">
	<INFORMATION format-rev="2" xsi:type="packageInfo">
		<ATTRIBUTES>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Type</NAME>
				<VALUE xsi:type="string">MANUAL</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Path</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Creation Date</NAME>
				<VALUE xsi:type="string">2020-06-04</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Status</NAME>
				<VALUE xsi:type="string">Design</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Estimated DevTime</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Execution Priority</NAME>
				<VALUE xsi:type="string">01-mandatory to be conducted</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Test Execution</NAME>
				<VALUE xsi:type="string">manual</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">KIFA Bereich</NAME>
				<VALUE xsi:type="string">ALL</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Designer</NAME>
				<VALUE xsi:type="string">qxi0247</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">TestInstance</NAME>
				<VALUE xsi:type="string">HIL_TSP_Antrieb_01</VALUE>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Individ. Field 1</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Model involved</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">OMR-TA</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">ECU</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Priority</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Regression relevant</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Design Organization</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
			<ATTRIBUTE xsi:type="attributeInst">
				<NAME xsi:type="string">Estimated Duration (min)</NAME>
				<VALUE xsi:type="string"/>
			</ATTRIBUTE>
		</ATTRIBUTES>
		<DESCRIPTION xsi:type="string">TSC_216:
The AE shall check that the FwgSigIn1 signal is within the valid range of [FwgSigIn1_Vmin, FwgSigIn1_VMax]  and mark it as &quot;invalid&quot; if the signal is out of range.

FTT &lt; 50 ms
Safe state: &quot;invalid&quot; signal FwgSig1_v

TSC_232:
The AE shall use the value of the FwgSig2_v as reference for the accelerator signal AccPos and limit the positive gradient of AccPos signal to a maximum AccPos_MaxGrad  if the FwgSig1_v is marked as invalid and the FwgSig2_v is marked as valid and the BrkAbs_v is not set.

Note: The controllability is only dependent on the FwgSig1_v and FwgSig2_v signals condition, the BrkAbs_v signal condition is only considered for comfort purposes.

FTT &lt; 50 ms
Safe state: Signal AccPos = FwgSig2_v and positive gradient limited to AccPos_MaxGrad  [%/s] and mark AccPos signal as &quot;fault&quot;

Zum Test wird:
 * Das applizierte Limit ermittelt und die FWG1 Spannung um 0.2V darunter gesetzt.
 * Die Reaktion von  RecuProhibitionAccelerator überwacht.
 * Die Reaktionszeit &lt; 50ms mittels Traceanalyse verifiziert.
 * Der Gradient &lt; 400% mittels Traceanalyse verifiziert

Erstellt: 26.05.2021 (Ralf Zeischka)</DESCRIPTION>
		<VERSION xsi:type="string">$Rev: 121 $</VERSION>
		<TAGS>
			<TAG xsi:type="string">TESTCASE</TAG>
		</TAGS>
		<ALTERNATE-CALL-REPRESENTATION-ACTION-FIELD-TEMPLATE xsi:type="string"/>
		<ALTERNATE-CALL-REPRESENTATION-EXPECTATION-FIELD-TEMPLATE xsi:type="string"/>
	</INFORMATION>
	<VARIABLES xsi:type="variableContainer">
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator2MaxLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">maxGradient</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator1MinLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">fwg2_v</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAccRef</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAcceleratorWert1</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">pctAcceleratorWert2</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">accelerator2MinLimit</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">trigg</NAME>
			<DESCRIPTION xsi:type="string"/>
			<DEFAULT-VALUE format-rev="1" xsi:type="value">
				<DATA xsi:type="integer">0</DATA>
				<TEXTDATA xsi:type="string">0</TEXTDATA>
			</DEFAULT-VALUE>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">startGradientenmessung</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
		<VARIABLE format-rev="1" xsi:type="variable">
			<NAME xsi:type="string">startGradientenmessung1</NAME>
			<DEFAULT-VALUE xsi:type="undefined"/>
		</VARIABLE>
	</VARIABLES>
	<MAPPING format-rev="1" xsi:type="localMappingSpace">
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_ElectricDriveReceiver.PctAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.PctAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</ID>
			<XACCESS xsi:type="xaMeasValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</LABEL>
				<RASTER xsi:type="string">Port-Default</RASTER>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_PctAcceleratorFailPositiveGradient_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator1MinLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator1MinLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator2MaxLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">DME1_DDE1/Sda_UAccelerator2MinLimit_P</ID>
			<XACCESS xsi:type="xaCalibValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<ECU-KEY xsi:type="string">DME1_DDE1</ECU-KEY>
				<LABEL xsi:type="string">Sda_UAccelerator2MinLimit_P</LABEL>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">Fahrbereitschaft</ID>
			<XACCESS format-rev="2" xsi:type="xaModelSignal">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/SGVM_MOTORRAD_AE/KOMBI/TA_BMS_AE_Fahrbereitschaft/Switch/Out1</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">Killswitch</ID>
			<DESCRIPTION xsi:type="string">Killswitch E_S_KILL; Ecu Pin 25; BOB B02</DESCRIPTION>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/INTERFACE_IO_BMS_AE/CTRL_IO_BMS/Killswitch_[0|1]/CTRL_E_S_KILL_[0|1]/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">StartTaster</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/INTERFACE_IO_BMS_AE/CTRL_IO_BMS_AE/CTRL_E_S_START_[0|1]/CTRL_E_S_START_[0|1]/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_BMS_AE_Fahrbereitschaft</ID>
			<XACCESS format-rev="2" xsi:type="xaModelSignal">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/SGVM_MOTORRAD_AE/KOMBI/TA_BMS_AE_Fahrbereitschaft/Switch/Out1</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
			<AUTO-GENERATED xsi:type="boolean">False</AUTO-GENERATED>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Active/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG1__V/Control/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Active/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
		<MAPPING-ITEM format-rev="2" xsi:type="mappingItem">
			<ID xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</ID>
			<XACCESS format-rev="2" xsi:type="xaModelValueVariable">
				<MAPPING-ENUM xsi:type="vtabInfoEmpty"/>
				<MODEL-KEY xsi:type="string">Implementierung</MODEL-KEY>
				<VARIABLE-PATH xsi:type="string">Model Root/Model_Root/INTERFACE_IO_BMS_AE/HWI_IO_BMS_AE/HWI_BMS_AE_SCLX/DAC/TA_HWI_E_A_FWG2__V/Control/Value</VARIABLE-PATH>
				<VARIABLE-TYPE xsi:type="string">flt(64,IEEE)*</VARIABLE-TYPE>
			</XACCESS>
		</MAPPING-ITEM>
	</MAPPING>
	<TM-INFO format-rev="1" xsi:type="testManagementInfo"/>
	<TRACE-RECORDING format-rev="2" xsi:type="recordingManager">
		<SIGNAL-GROUPS>
			<ELEMENT format-rev="2" xsi:type="signalGroup">
				<NAME xsi:type="string">Gradientenmessung 1</NAME>
				<DESCRIPTION xsi:type="string"/>
				<SIGNALS>
					<ELEMENT xsi:type="signal">
						<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</MAPPING-REF>
					</ELEMENT>
				</SIGNALS>
			</ELEMENT>
		</SIGNAL-GROUPS>
		<RECORDING-GROUP>
			<ELEMENT format-rev="3" xsi:type="recordingGroup">
				<NAME xsi:type="string">Aufnahmegruppe von Gradientenmessung 1</NAME>
				<UUID xsi:type="string">31d6ef18a15a11eb849bc8d9d205aa05</UUID>
				<SIGNAL-GROUP-REF>Gradientenmessung 1</SIGNAL-GROUP-REF>
			</ELEMENT>
		</RECORDING-GROUP>
		<SYNCHRONISATION-CONFIG xsi:type="syncConfig"/>
	</TRACE-RECORDING>
	<TESTSTEPS xsi:type="testCase">
		<TESTSTEP format-rev="3" id="6543b0e9-2e1b-427a-a88f-65d2d9486fa4" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Precondition</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP format-rev="2" id="27995146-601b-4152-bfcc-1ca80dcf0f77" xsi:type="tsPackage">
				<PACKAGE-REFERENCE format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="string">..\\..\\lib\\HiL Init.pkg</VALUE>
				</PACKAGE-REFERENCE>
				<PARAM-ASSIGNMENTS/>
				<ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="94da3a57-84c2-4ef5-83c8-9b3358e909b9" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Ausgangszustand pruefen (FwgSigIn1_Vmin, FwgSigIn1_VMax lesen)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="8d1a2e0d-8ed4-4d19-bb05-7e3ddd4adcf2" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.0</VALUE>
						</VALUE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="4fcd71fb-3649-435f-83c7-09c321dc48d7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator1MinLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator1MinLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.7</VALUE>
						</VALUE>
						<TOLERANCE style="percentage" xsi:type="Tolerance">
							<VALUE format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="integer">1</VALUE>
							</VALUE>
						</TOLERANCE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="3c42a7f4-a44f-45b3-a29b-1e58e75073d0" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MaxLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator2MaxLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="4daf152f-7bc8-423a-a060-a3f708aa46f7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_UAccelerator2MinLimit_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">accelerator2MinLimit</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<EXPECTATION xsi:type="timelessOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.3</VALUE>
						</VALUE>
						<TOLERANCE style="absolute-value" xsi:type="Tolerance">
							<VALUE format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="float">0.01</VALUE>
							</VALUE>
						</TOLERANCE>
					</EXPRESSION>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_v</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="ad6ec1c3-f4e1-4062-9a76-7793bc5ac57a" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/Sda_PctAcceleratorFailPositiveGradient_P</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">maxGradient</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_undefined_perc/s</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="9222e164-15da-490e-9b7e-f8c45ba6efe7" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">fwg2_v</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="1c982b9b-f376-4b21-a169-964603d516ac" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="2191552f-c313-4073-9779-c227cdfb7341" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern  (FwgSigIn1 &lt; FwgSigIn1_VMin setzen)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="8e661123-05d6-40dc-ba61-2a73523f15e2" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">1</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="24292a7b-03fe-4612-884a-ed9107c7423e" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">accelerator1MinLimit</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.2</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="4c09ee91-fe69-499c-ab57-c8329a9d84e8" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion pruefen (Safe state: &quot;invalid&quot; signal FwgSig1_v)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="a3498426-8406-4a6d-b6ef-95ecc465766e" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">1.0</VALUE>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="a7dc905b-3809-4f0e-881a-15cf583029d8" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">pctAccRef</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="71108f35-5f48-4de5-a74a-4d9ce6e163a8" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VARIABLE-REFS>
					<VARIABLE-NAME dkey="default">
						<DVALUE xsi:type="string">fwg2_v</DVALUE>
					</VARIABLE-NAME>
				</VARIABLE-REFS>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="e31e741a-0e42-4cd3-a74a-d1938b3a9a81" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="10f2dbd2-afd6-4447-9ee7-960573211274" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern (FWG2 manipulieren)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="0d7f02f8-3a46-4677-86be-b8626587b16f" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">1</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="593bbac2-bcd2-4f1e-8e5b-cf5b0af710f5" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_ADD</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">fwg2_v</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.2</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="2f2e0c03-6bb9-428b-9538-14174b909b12" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">500</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="6ea0cb9d-d1f3-43b9-a22a-2758f3f068c1" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion pruefen (PctAcceleration folgt FWG2)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="f292540a-b0e2-4cd2-a8a5-6416f49d8d59" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.RecuProhibitionAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">==</RELATION>
						<VALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">1.0</VALUE>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="18fc5c8f-3614-4cec-906b-d47d89ff152f" xsi:type="tsRead">
				<MAPPING-REF xsi:type="string">DME1_DDE1/CPS_SafetyElectricDriveReceiver.PctAccelerator</MAPPING-REF>
				<EXPECTATION xsi:type="finallyTrueOption">
					<EXPRESSION xsi:type="builtNumericExpression">
						<RELATION xsi:type="string">&gt;=</RELATION>
						<VALUE xsi:type="binaryOpBaseExpression">
							<NAME xsi:type="string">BINARY_ADD</NAME>
							<FIRST-COMPONENT xsi:type="varBaseExpression">
								<NAME xsi:type="string">pctAccRef</NAME>
							</FIRST-COMPONENT>
							<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
								<VALUE xsi:type="integer">2</VALUE>
							</SECOND-COMPONENT>
						</VALUE>
					</EXPRESSION>
					<TIME format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">500</VALUE>
					</TIME>
				</EXPECTATION>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<Z-UNIT xsi:type="string">u_none</Z-UNIT>
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="ffd6106d-c36e-4ee6-bc59-46d752c91809" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zustand aendern (FWG2 Max Wert zur Gradientenbestimmung)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="e963700f-2941-422f-b29e-546f81143876" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="binaryOpBaseExpression">
						<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
						<FIRST-COMPONENT xsi:type="varBaseExpression">
							<NAME xsi:type="string">accelerator2MaxLimit</NAME>
						</FIRST-COMPONENT>
						<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="float">0.1</VALUE>
						</SECOND-COMPONENT>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="8619f375-7ee0-4f32-b2a9-b0eb54545ab2" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">50</VALUE>
				</TIME>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="e96ac323-9a18-442b-b14f-d74d0db4e85b" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Reaktion prüfen (Gradientenmessung per Traceanalyse)</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="86ff1301-ed56-40ea-b054-a9ba24c0409e" name="TsStartTrace" xsi:type="utility-9308d99e-50ac-11dc-8fe5-001143176a1c">
				<NAME xsi:type="string">Gradientenmessung</NAME>
				<RECORDING-GROUP-REF-BY-UUID>31d6ef18a15a11eb849bc8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
			</TESTSTEP>
			<TESTSTEP id="7afd0f0b-f0e1-4137-a2a7-0ca95957fbb2" name="TsWait" xsi:type="utility-62d5a961-4fef-11dc-9944-0013728784ee">
				<TIME format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="integer">250</VALUE>
				</TIME>
			</TESTSTEP>
			<TESTSTEP id="c1a71aa1-f70f-498b-af0f-dc68fc42e0a4" name="TsStopTrace" xsi:type="utility-d12f5791-50ac-11dc-8eb4-001143176a1c">
				<NAME xsi:type="string">Gradientenmessung</NAME>
				<RECORDING-GROUP-REF-BY-UUID>31d6ef18a15a11eb849bc8d9d205aa05</RECORDING-GROUP-REF-BY-UUID>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="735fe1b1-7860-4c13-9c6b-db86d1d85793" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Zuruecksetzen</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP id="f8403c81-9cae-4e33-9fe8-5956e4ec1920" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="1fef7048-a519-437e-8d08-238ce22729e2" xsi:type="tsRestore">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
			</TESTSTEP>
			<TESTSTEP id="9e421a84-a888-4efb-ba8b-03841e990955" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator1MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="b48a4a42-e09d-4055-b9d6-1605e11dfaf8" xsi:type="tsWrite">
				<ENABLED xsi:type="boolean">False</ENABLED>
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="integer">0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="bad88484-dcc8-4f11-9db0-3c72e4c69bd1" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="float">0.0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="cdb13ac1-c3ab-438f-bd4f-7887835b891a" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Active/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA format-rev="1" xsi:type="valueBaseExpression">
						<VALUE xsi:type="float">0.0</VALUE>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="810076c3-0c38-40f0-9bb4-78a6c87d1aec" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG1__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator1MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
			<TESTSTEP id="e5bd43d1-216f-4fa6-9c1e-def807257043" xsi:type="tsWrite">
				<MAPPING-REF xsi:type="string">TA_HWI_E_A_FWG2__V/Control/Value</MAPPING-REF>
				<VALUE xsi:type="expressionValue">
					<DATA xsi:type="varBaseExpression">
						<NAME xsi:type="string">accelerator2MinLimit</NAME>
					</DATA>
				</VALUE>
				<METRIC format-rev="1" xsi:type="metricInfo">
					<VALUE-TYPE xsi:type="string">PHYS</VALUE-TYPE>
					<DATA-TYPE xsi:type="string">VALUE</DATA-TYPE>
				</METRIC>
			</TESTSTEP>
		</TESTSTEP>
		<TESTSTEP format-rev="3" id="326a0c85-3825-4a95-9d2b-f8447b454835" name="TsBlock" xsi:type="utility-2752ad1e-4fef-11dc-81d4-0013728784ee">
			<ACTION xsi:type="I18NItem">
				<MULTILANGDATA>
					<ELEMENT dkey="de_DE">
						<DVALUE xsi:type="string">Postcondition</DVALUE>
					</ELEMENT>
				</MULTILANGDATA>
				<INITIAL-LANGUAGE xsi:type="string">de_DE</INITIAL-LANGUAGE>
			</ACTION>
			<TESTSTEP format-rev="2" id="6b7f756b-ac87-4ad4-bca4-3b6a69452d61" xsi:type="tsPackage">
				<PACKAGE-REFERENCE format-rev="1" xsi:type="valueBaseExpression">
					<VALUE xsi:type="string">C:\\HIL_AE\\Packages\\Packages\\lib\\TerminateExecution.pkg</VALUE>
				</PACKAGE-REFERENCE>
				<PARAM-ASSIGNMENTS>
					<ASSIGNMENT dkey="Fahrzeug_stillstand">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Fehlerspeicher">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Klemmenwechsel_KL15">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">False</VALUE>
						</DVALUE>
					</ASSIGNMENT>
					<ASSIGNMENT dkey="Klemmenwechsel_KL30">
						<DVALUE format-rev="1" xsi:type="valueBaseExpression">
							<VALUE xsi:type="string">True</VALUE>
						</DVALUE>
					</ASSIGNMENT>
				</PARAM-ASSIGNMENTS>
				<ALTERNATIVE-MAPPING-SPACE format-rev="1" xsi:type="mappingSpace"/>
			</TESTSTEP>
		</TESTSTEP>
	</TESTSTEPS>
	<TRACE-ANALYSES format-rev="1" xsi:type="traceAnalyses">
		<TRACE-ANALYSIS format-rev="3" uuid="0f91da22a14c11eb9cf5c8d9d205aa05" xsi:type="traceAnalysis">
			<NAME xsi:type="string">Gradientenmessung</NAME>
			<ANALYSISITEM format-rev="1" uuid="0f92014ca14c11ebbf92c8d9d205aa05" xsi:type="episode">
				<NAME xsi:type="string">Neue Episode</NAME>
				<ANALYSISITEM format-rev="2" poolRev="6" uuid="983e2637a3a211eb9607c8d9d205aa05" xsi:type="referencingTraceStep">
					<NAME xsi:type="string">Gradient_Zeitfenster</NAME>
					<SIGNAL-BINDING xsi:type="signalBinding">
						<PARAMETER dkey="messwert">
							<DVALUE format-rev="1" xsi:type="signalParameter">
								<REF-SIGNAL-NAME xsi:type="string">pctAccelerator</REF-SIGNAL-NAME>
							</DVALUE>
						</PARAMETER>
					</SIGNAL-BINDING>
					<PARAM-ASSIGNMENTS xsi:type="trcpParameterAssignments">
						<PARAM-ASSIGNMENTS>
							<ASSIGNMENT dkey="MaxGradient">
								<DVALUE xsi:type="binaryOpBaseExpression">
									<NAME xsi:type="string">BINARY_ADD</NAME>
									<FIRST-COMPONENT xsi:type="varBaseExpression">
										<NAME xsi:type="string">maxGradient</NAME>
									</FIRST-COMPONENT>
									<SECOND-COMPONENT xsi:type="binaryOpBaseExpression">
										<NAME xsi:type="string">BINARY_MULTIPLY</NAME>
										<FIRST-COMPONENT xsi:type="varBaseExpression">
											<NAME xsi:type="string">maxGradient</NAME>
										</FIRST-COMPONENT>
										<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
											<VALUE xsi:type="float">0.05</VALUE>
										</SECOND-COMPONENT>
									</SECOND-COMPONENT>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="MinGradient">
								<DVALUE xsi:type="binaryOpBaseExpression">
									<NAME xsi:type="string">BINARY_SUBTRACT</NAME>
									<FIRST-COMPONENT xsi:type="varBaseExpression">
										<NAME xsi:type="string">maxGradient</NAME>
									</FIRST-COMPONENT>
									<SECOND-COMPONENT xsi:type="binaryOpBaseExpression">
										<NAME xsi:type="string">BINARY_MULTIPLY</NAME>
										<FIRST-COMPONENT xsi:type="varBaseExpression">
											<NAME xsi:type="string">maxGradient</NAME>
										</FIRST-COMPONENT>
										<SECOND-COMPONENT format-rev="1" xsi:type="valueBaseExpression">
											<VALUE xsi:type="float">0.05</VALUE>
										</SECOND-COMPONENT>
									</SECOND-COMPONENT>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="Startzeit">
								<DVALUE format-rev="1" xsi:type="valueBaseExpression">
									<VALUE xsi:type="integer">0</VALUE>
								</DVALUE>
							</ASSIGNMENT>
							<ASSIGNMENT dkey="Stopzeit">
								<DVALUE format-rev="1" xsi:type="valueBaseExpression">
									<VALUE xsi:type="integer">500</VALUE>
								</DVALUE>
							</ASSIGNMENT>
						</PARAM-ASSIGNMENTS>
					</PARAM-ASSIGNMENTS>
					<REPORT-CONFIG xsi:type="trcpReportConfig">
						<MAX-SPOTS xsi:type="integer">100</MAX-SPOTS>
						<MIN-DELTA xsi:type="float">0.0</MIN-DELTA>
					</REPORT-CONFIG>
					<PROTOTYPE-ID xsi:type="string">Gradient_Zeitfenster</PROTOTYPE-ID>
				</ANALYSISITEM>
			</ANALYSISITEM>
			<VIRTUAL-SIGNALS-MANAGER xsi:type="virtualSignalsManager">
				<VIRTUAL-SIGNALS>
					<VIRTUAL-SIGNAL uuid="5cef80faa14c11eb983dc8d9d205aa05" xsi:type="virtualSessionSignal">
						<NAME xsi:type="string">pctAccelerator</NAME>
						<SIGNAL-FOR-ANALYSIS xsi:type="recordingSignalForAnalysis">
							<KEY xsi:type="string">DME1_DDE1/CPS_ElectricDriveReceiver.PctAccelerator</KEY>
							<SOURCE-TYPE xsi:type="string">RECORDING-GROUP</SOURCE-TYPE>
							<SOURCE-REF xsi:type="string">31d6ef18a15a11eb849bc8d9d205aa05</SOURCE-REF>
						</SIGNAL-FOR-ANALYSIS>
						<EVENT-TYPE xsi:type="string">PHY</EVENT-TYPE>
					</VIRTUAL-SIGNAL>
				</VIRTUAL-SIGNALS>
			</VIRTUAL-SIGNALS-MANAGER>
		</TRACE-ANALYSIS>
	</TRACE-ANALYSES>
</PACKAGE>
"""

# # import xml.etree.ElementTree as ET
# # # tree = ET.fromstring(x)
# # tree = ET.parse('test.xml')
# # root = tree.getroot()
# # print(root)
# # lol = root[0]
# # print(lol)
# # # lol = lol[0]
# # # print(lol)
# # for cc in lol:
# #     print(f'\tELEMENT!!')
# #     print(cc[0])
#
# # for cc in root:
# #     for ccc in cc:
# #         print(ccc)
# #         for cccc in ccc:
# #             print(f'\t{cccc}')
# #     print(cc)
#
#
# import re
# x = x
# x = re.sub('&gt;', '>', x)
# x = re.sub('&lt;', '>', x)
# x = x.replace('TESTSTEPS', 'Simon was here')  # split funktioniert sonst nicht
# split = re.split('<TESTSTEP', x)
# split = split[1:]
# lel = []
#
#
# def get_values(ss):
#     e = re.findall(r'(?<=>)(.*)(?=</VALUE>)', ss)
#     return e
#
#
# def get_inside(ss):
#     e = re.findall(r'(?<=>)(.*)(?=</)', ss)
#     return e
#
#
# def split_and_get(ss: str, get_after, get_optional=None):
#     """get_after='binaryOpBaseExpression', get_optional='FIRST-COMPONENT'"""
#     e = ss.split(get_after)
#     e = e[1]
#     if get_optional:
#         e = e.split(get_optional)
#         e = e[1]
#     e = get_inside(e)
#     return e
#
#
# def get_expression(ss):
#     pass
#
#
# for i, ee in enumerate(split):
#     s = ''
#     toomanyhits = 0
#
#     if '<ENABLED ' in ee:
#         continue  # Ignoring commented lines
#     elif '<TIME ' in ee:
#         # 'name="TsWait"' in ee:
#         toomanyhits += 1
#         continue  # ignore wait blocks
#     # elif 'xsi:type="tsRestore"' in ee:
#     #     continue  # Ignoring resets?
#
#     elif 'name="TsStartTrace"' in ee:
#         s = f'{s}Start Trace (SFEH)'
#     elif 'name="TsStopTrace"' in ee:
#         # if 'RECORDING-GROUP-REF-BY-UUID' in ee:
#         s = f'{s}Stop Trace (SFEH)'
#
#     elif '<ACTION xsi:type="I18NItem">' in ee:
#         # Kommentarblock
#         # '<ELEMENT dkey="de_DE">' in ee:
#         e = re.split(r'(<DVALUE xsi:type="string">)', ee)
#         e = e[2]
#         e = re.split(r'<\/', e)[0]
#         s = blue_string(f'{s}{e}')
#
#     elif '<PACKAGE-REFERENCE' in ee:
#         # C:/HIL_AE/Packages/Packages/FuSi/EATR/../../lib/HiL Init.pkg
#         e = re.split(r'<VALUE xsi:type="string">', ee)
#         e = e[1]
#         e = re.split(r'</VALUE>', e)[0]
#         if 'HiL Init.pkg' in e or 'TerminateExecution.pkg' in e:
#             continue
#         else:
#             s = f'{s}>>{BColors.HEADER}{e}{BColors.RESET}'
#
#     elif '<MAPPING-REF' in ee:
#         e_name = split_and_get(ee, 'MAPPING-REF')[0]  # if '/Active/Value' in e_name:
#         e3 = get_inside(ee)
#         # if 'binaryOpBaseExpression>' in ee:
#         #     b = split_and_get(ee, get_after='binaryOpBaseExpression')[0]
#         #     b = b.replace('BINARY_ADD', '+')
#         #     b = b.replace('BINARY_SUBTRACT', '-')
#         #     a = split_and_get(ee, get_after='binaryOpBaseExpression', get_optional='FIRST-COMPONENT')[0]
#         #     c = split_and_get(ee, get_after='binaryOpBaseExpression', get_optional='SECOND-COMPONENT')[0]
#         #     e3 = f'({a} {b} {c})'
#         # else:
#         #     e3 = get_inside(ee)
#         #     e3 = e3[1]
#
#         if '"tsWrite"' in ee:
#             # s = s.replace('\t', '')  # force one tab only, in the following line
#             s = f'WRITE:{e_name}\t:={e}'
#
#         elif '"tsRead"' in ee:
#
#             if 'EXPECTATION' in ee:
#
#                 if 'TOLERANCE' in ee:
#                     e = split_and_get(ee, 'TOLERANCE')[0]
#                     e_toleranz = f' (+-{e})'
#                 else:
#                     e_toleranz = ''
#
#                 if '<TIME' in ee:
#                     e = split_and_get(ee, '<TIME')[0]
#                     e_time = f' (Zeit {e})'
#                 else:
#                     e_time = ''
#
#                 e_rel = re.findall(r'(?<=<RELATION xsi:type="string">)(.*)(?=</)', ee)[0]
#                 e_val = split_and_get(ee, 'valueBaseExpression')[0]
#                 e3 = f'{BColors.GREEN}{e_rel} {e}{e_toleranz}{e_time}{BColors.RESET_COLOR}'
#             else:
#                 e3 = ''
#
#             if 'VARIABLE-NAME' in ee:
#                 opt_saveasvar = re.findall(r'(?<=<DVALUE xsi:type="string">)(.*)(?=</DVALUE>)', ee)
#                 opt_saveasvar = f' -> {BColors.ITALIC}{opt_saveasvar[0]}{BColors.RESET}'
#             else:
#                 opt_saveasvar = ''
#             s = f'{e_name}\t{e3}{opt_saveasvar}'
#         elif '"tsRestore"' in ee:
#             continue
#         else:
#             raise NotImplementedError
#
#     else:
#         raise NotImplementedError
#
#     s = s.replace('/Active/Value', '')
#     s = s.replace('/Control/Value', '')
#     s = s.replace('DME1_DDE1/', '')
#     s = s.replace('DME1_DDE1', '')
#     print(s)
#     # lel.append(s)
#
# lel = '\n'.join(lel)
# print(lel)

# from bs4 import BeautifulSoup
#
# # Reading the data inside the xml
# # file to a variable under the name
# # data
# with open('bmw-motorrad-delete-xml.xml', 'r') as f:
#     data = f.read()
#
# # Passing the stored data inside
# # the beautifulsoup parser, storing
# # the returned object
# Bs_data = BeautifulSoup(data, "xml")
#
# # Finding all instances of tag
# # `unique`
# b_unique = Bs_data.find_all('unique')
#
# print(b_unique)
#
# # Using find() to extract attributes
# # of the first instance of the tag
# b_name = Bs_data.find('child', {'name': 'Frank'})
#
# print(b_name)
#
# # Extracting the data stored in a
# # specific attribute of the
# # `child` tag
# value = b_name.get('test')
#
# print(value)


import xmltodict


def get_all_teststeps(ee):

    if isinstance(ee, list):
        for ii, x in enumerate(ee):
            return get_all_teststeps(x)
    elif isinstance(ee, dict):
        name = ee.get('@name')
        action = ee.get('ACTION')
        enabled = ee.get('ENABLED')
        time = ee.get('TIME')
        action = ee.get('ACTION')
        xsi_type = ee.get('@xsi:type')
        mapping_ref = ee.get('MAPPING-REF')
        package_reference = ee.get('PACKAGE-REFERENCE')
        teststep = ee.get('TESTSTEP')
        expression = ee.get('EXPRESSION')
        tolerance = ee.get('TOLERANCE')
        expr = ee.get('EXPRESSION')
        dvalue = ee.get('DVALUE')
        default_value = ee.get('DEFAULT-VALUE')
        element = ee.get('ELEMENT')
        value = ee.get('value')
        if name == 'TsBlock':
            action = action['MULTILANGDATA']['ELEMENT']['DVALUE']['#text']
            print(action)
            for t in teststep:
                get_all_teststeps(t)
        if package_reference:
            package_reference = package_reference['PATH-EXPRESSION']['VALUE']['#text']
            param_assignments = package_reference['PARAM-ASSIGNMENTS']['ASSIGNMENT']
            print(package_reference, param_assignments)
        # if element:
        #     for x in element:
        #         get_all_teststeps(x)




    else:
        raise NotImplementedError


hohoho = xmltodict.parse(x)
eix_list = hohoho['EIX-LIST']
ee = eix_list['ITEMS']
ee = ee['ELEMENT']
for elf in ee:


    if t:
        asd = get_all_teststeps(t)
        print(asd)


'ENABLED': '#text': # 'False'
'@name': 'TsBlock'
'@name': 'TsWait'
'TIME': 'VALUE': '#text': # 1000
'ACTION': 'MULTILANGDATA': 'ELEMENT': 'DVALUE': '#text': # precondition
'TESTSTEP': list
'@xsi:type'
    'MAPPING-REF': '#text'
    'VALUE': 'VALUE':
    'VALUE': 'DATA': 'VALUE':
    'EXPECTATION': 'timelessOption'
        '@xsi:type'
        'EXPRESSION':
            'RELATION': '#text'
            '@xsi:type' 'valueBaseExpression'
            'VALUE': 'VALUE': '#text'
            'TOLERANCE':
                '@style': # percentage
                'VALUE':
                    'VALUE': '#text'
    'VARIABLE-REFS': 'VARIABLE-NAME': 'DVALUE': '#text'
    'METRIC'



