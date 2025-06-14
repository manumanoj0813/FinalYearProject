import React, { useState, useEffect } from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  Heading,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useToast,
  Button,
  HStack,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Avatar,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaUser, FaChevronDown, FaSignOutAlt, FaCog } from 'react-icons/fa';

import { AudioRecorder } from './components/AudioRecorder';
import { AnalysisDisplay } from './components/AnalysisDisplay';
import { ProgressDashboard } from './components/ProgressDashboard';
import { PracticeSession } from './components/PracticeSession';
import { EnhancedAnalysis } from './components/EnhancedAnalysis';
import { ComparisonCharts } from './components/ComparisonCharts';
import { LoginForm } from './components/LoginForm';
import { ThemeToggle } from './components/ThemeToggle';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { VoiceAnalysis } from './types';
import { theme } from './theme';

const UserMenu: React.FC = () => {
  const { user, logout } = useAuth();
  
  return (
    <Menu>
      <MenuButton
        as={Button}
        rightIcon={<FaChevronDown />}
        leftIcon={<Avatar size="xs" name={user?.username} />}
        variant="outline"
      >
        {user?.username}
      </MenuButton>
      <MenuList>
        <MenuItem icon={<FaCog />}>Settings</MenuItem>
        <MenuItem icon={<FaSignOutAlt />} onClick={logout}>
          Logout
        </MenuItem>
      </MenuList>
    </Menu>
  );
};

const MainContent: React.FC = () => {
  const [analysis, setAnalysis] = useState<VoiceAnalysis | null>(null);
  const { user, isAuthenticated } = useAuth();
  const toast = useToast();
  const [activeTab, setActiveTab] = useState(0);

  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const textColor = useColorModeValue('gray.600', 'gray.300');

  const handleAnalysisComplete = (newAnalysis: VoiceAnalysis) => {
    setAnalysis(newAnalysis);
    const currentSessions = Number(localStorage.getItem('totalSessions') || '0');
    localStorage.setItem('totalSessions', (currentSessions + 1).toString());

    toast({
      title: 'Analysis Complete',
      description: 'Your voice analysis is ready!',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  if (!isAuthenticated) {
    return (
      <Box minH="100vh" py={8}>
        <VStack spacing={8} maxW="1200px" mx="auto" px={4}>
          <Heading size="2xl" color="purple.600">
            Vocal IQ
          </Heading>
          <Text fontSize="xl" color={textColor} textAlign="center" mb={8}>
            AI-Powered Voice Analytics for Smarter Learning
          </Text>
          <LoginForm />
        </VStack>
      </Box>
    );
  }

  return (
    <Box minH="100vh" py={8}>
      <VStack spacing={8} maxW="1200px" mx="auto" px={4}>
        <HStack w="full" justify="space-between">
          <Heading size="2xl" color="purple.600">
            Vocal IQ
          </Heading>
          <HStack spacing={4}>
            <ThemeToggle />
            <UserMenu />
          </HStack>
        </HStack>

        <Text fontSize="xl" color={textColor} textAlign="center">
          Welcome back, {user?.username}! Ready to improve your speaking skills?
        </Text>

        <Tabs 
          variant="soft-rounded" 
          colorScheme="purple" 
          w="full" 
          index={activeTab}
          onChange={setActiveTab}
        >
          <TabList justifyContent="center" mb={8} flexWrap="wrap">
            <Tab>Quick Analysis</Tab>
            <Tab>Enhanced Analysis</Tab>
            <Tab>Practice Session</Tab>
            <Tab>Comparison Charts</Tab>
            <Tab>Progress Dashboard</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <VStack spacing={8} w="full">
                <Box w="full" maxW="600px" mx="auto">
                  <AudioRecorder
                    onAnalysisComplete={handleAnalysisComplete}
                    sessionType="quick"
                    topic="general"
                  />
                </Box>
                {analysis && <AnalysisDisplay analysis={analysis} />}
              </VStack>
            </TabPanel>

            <TabPanel>
              <VStack spacing={8} w="full">
                <Box w="full" maxW="600px" mx="auto">
                  <AudioRecorder
                    onAnalysisComplete={handleAnalysisComplete}
                    sessionType="enhanced"
                    topic="general"
                  />
                </Box>
                {analysis && <EnhancedAnalysis analysis={analysis} />}
              </VStack>
            </TabPanel>

            <TabPanel>
              <PracticeSession onAnalysisComplete={handleAnalysisComplete} />
            </TabPanel>

            <TabPanel>
              <ComparisonCharts />
            </TabPanel>

            <TabPanel>
              <ProgressDashboard />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
};

const App: React.FC = () => {
  return (
    <ChakraProvider theme={theme}>
      <ThemeProvider>
        <AuthProvider>
          <MainContent />
        </AuthProvider>
      </ThemeProvider>
    </ChakraProvider>
  );
};

export default App;
